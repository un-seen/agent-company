import re
import redis
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import logging

import redis.exceptions
from agentcompany.driver.dict import dict_rows_to_markdown_table
from agentcompany.mcp.utils import truncate_content
from agentcompany.extensions.environments.exceptions import InterpreterError
from agentcompany.extensions.environments.base import ExecutionEnvironment

logger = logging.getLogger(__name__)

BASE_BUILTIN_MODULES = ["time", "unicodedata"]
DEFAULT_MAX_LEN_OUTPUT = 50000


class RedisRedisInterpreter(ExecutionEnvironment):
    
    language: str = "redis"
    
    def __init__(
        self,
        session_id: str,
        mcp_servers: Dict,
        host: str,
        port: int,
        dbname: int,
        password: Optional[str] = None,
        additional_authorized_imports: Optional[List[str]] = None
    ):
        self.custom_tools = {}
        self.state = {}
        self.max_print_outputs_length = DEFAULT_MAX_LEN_OUTPUT
        self.additional_authorized_imports = additional_authorized_imports or []
        
        # Redis configuration
        self.redis_config = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "password": password
        }
        redis_url = "redis://"
        if password:
            redis_url += f":{password}@"
        redis_url += f"{host}:{port}/{dbname}"
        self.redis_client = redis.Redis.from_url(redis_url)
        logger.info(f"Connected to Redis at {host}:{port}, database {dbname}")
        
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.static_tools = mcp_servers
        self.storage = {}  # Stores step_id to Redis key mapping
        super().__init__(session_id=session_id, mcp_servers=mcp_servers)

    def reset_connection(self):
        self.redis_client.close()
        redis_url = "redis://"
        if self.redis_config["password"]:
            redis_url += f":{self.redis_config['password']}@"
        redis_url += f"{self.redis_config['host']}:{self.redis_config['port']}/{self.redis_config['dbname']}"
        self.redis_client = redis.Redis.from_url(redis_url)

    def get_hint(self, code_action: str) -> str:
        return ""
    
    def __call__(self, code_action: str, additional_variables: Dict, return_type: str = "string") -> Tuple[str, str, bool]:
        self.state.update(additional_variables)
        self.reset_connection()
        code_action = code_action.strip()
        if not code_action:
            raise InterpreterError("Empty Redis command")
        command_parts = code_action.split()
        command = command_parts[0].upper()
        args = command_parts[1:] if len(command_parts) > 1 else []
        try:
            response = self.redis_client.execute_command(command, *args)
        except redis.exceptions.RedisError as e:
            error_msg = f"Redis command '{command}' failed: {str(e)}"
            logger.error(error_msg)
            raise InterpreterError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error executing Redis command: {str(e)}"
            logger.error(error_msg)
            raise InterpreterError(error_msg)
        processed_rows = self._process_redis_response(response)
        markdown_table = dict_rows_to_markdown_table(processed_rows)
        logs = self.state.get("print_outputs", "")
        return markdown_table, logs, False

    def parse_function(self, code_blob: str) -> Dict[str, Callable]:
        raise NotImplementedError("parse_function not implemented")
    
    def _process_redis_response(self, response):
        # TODO return dataframe
        if isinstance(response, bytes):
            response = response.decode('utf-8')
        if isinstance(response, dict):
            return [{'field': k, 'value': v} for k, v in response.items()]
        elif isinstance(response, list):
            processed = []
            for idx, item in enumerate(response):
                if isinstance(item, bytes):
                    item = item.decode('utf-8')
                processed.append({'index': idx, 'value': item})
            return processed
        elif isinstance(response, (str, int, float)):
            return [{'value': response}]
        else:
            return [{'result': str(response)}]

    def attach_variables(self, variables: dict):
        self.state.update(variables)

    def parse_code_blob(self, code_blob: str) -> str:
        if re.match(r'^[A-Z]+\b', code_blob.strip(), re.IGNORECASE):
            return code_blob.strip()
        else:
            raise ValueError(f"No valid Redis code found: {code_blob}")
        

    def get_storage_id(self, next_step_id: int) -> str:
        return f"{self.session_id}_temp_{next_step_id}"

    def set_storage(self, next_step_id: int, code_action: str):
        temp_key = self.get_storage_id(next_step_id)
        try:
            response = self.redis_client.execute_command(*code_action.strip().split())
        except redis.exceptions.RedisError as e:
            raise InterpreterError(f"Failed to execute storage command: {str(e)}")
        
        if isinstance(response, dict):
            self.redis_client.hmset(temp_key, response)
            storage_type = 'hash'
        elif isinstance(response, list):
            self.redis_client.delete(temp_key)
            if response:
                self.redis_client.rpush(temp_key, *response)
            storage_type = 'list'
        elif isinstance(response, (bytes, str)):
            self.redis_client.set(temp_key, response)
            storage_type = 'string'
        elif isinstance(response, int):
            self.redis_client.set(temp_key, response)
            storage_type = 'integer'
        else:
            self.redis_client.set(temp_key, str(response))
            storage_type = 'unknown'
        
        self.storage[next_step_id] = {'key': temp_key, 'type': storage_type}

    def get_storage(self, next_step_id: int) -> str:
        if next_step_id not in self.storage:
            return ""
        storage_info = self.storage[next_step_id]
        temp_key = storage_info['key']
        storage_type = storage_info['type']
        info, example_cmds = [], []
        
        if storage_type == 'hash':
            example_cmds.append(f"HGETALL {temp_key}")
            fields = self.redis_client.hkeys(temp_key)
            if fields:
                info.append("Fields:")
                info.extend(f"- {field.decode()}" for field in fields)
        elif storage_type == 'list':
            example_cmds.append(f"LRANGE {temp_key} 0 -1")
            length = self.redis_client.llen(temp_key)
            info.append(f"List length: {length}")
            if length > 0:
                elements = self.redis_client.lrange(temp_key, 0, 2)
                info.append("Sample elements:")
                info.extend(f"{idx}: {elem.decode()}" for idx, elem in enumerate(elements))
        elif storage_type in ('string', 'integer'):
            example_cmds.append(f"GET {temp_key}")
            value = self.redis_client.get(temp_key)
            if value is not None:
                info.append(f"Value: {truncate_content(value.decode())}")
        else:
            example_cmds.append(f"GET {temp_key}")
            info.append("Data type: unknown")
        
        example_code = "\n".join(example_cmds)
        info_str = "\n".join(info)
        return (
            f"Data from step {next_step_id} stored in key '{temp_key}' (type: {storage_type}).\n"
            f"Example commands:\n{example_code}\nAdditional Info:\n{info_str}"
        )

    def get_final_storage(self) -> pd.DataFrame:
        if not self.storage:
            return pd.DataFrame()
        max_step_id = max(self.storage.keys())
        storage_info = self.storage[max_step_id]
        temp_key = storage_info['key']
        storage_type = storage_info['type']
        
        if storage_type == 'hash':
            data = self.redis_client.hgetall(temp_key)
            data = {k.decode(): v.decode() for k, v in data.items()}
            df = pd.DataFrame([data])
        elif storage_type == 'list':
            data = self.redis_client.lrange(temp_key, 0, -1)
            data = [item.decode() for item in data]
            df = pd.DataFrame(data, columns=['value'])
        elif storage_type in ('string', 'integer'):
            value = self.redis_client.get(temp_key)
            df = pd.DataFrame([{'value': value.decode()}]) if value else pd.DataFrame()
        else:
            df = pd.DataFrame()
        return df

    def reset_storage(self):
        for step_id in self.storage:
            temp_key = self.get_storage_id(step_id)
            self.redis_client.delete(temp_key)
        self.storage = {}