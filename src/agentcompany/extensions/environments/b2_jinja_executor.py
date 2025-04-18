import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
import boto3
from jinja2 import Template, Environment, BaseLoader, StrictUndefined, TemplateSyntaxError, TemplateRuntimeError
from botocore.client import Config
from agentcompany.extensions.environments.exceptions import InterpreterError
from agentcompany.extensions.environments.base import ExecutionEnvironment
from agentcompany.mcp.base import ModelContextProtocolImpl

logger = logging.getLogger(__name__)

S3_RESOURCE = None

def init_s3(endpoint_url, access_key_id, secret_access_key):
    global S3_RESOURCE
    if not S3_RESOURCE:
        S3_RESOURCE = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4"),
        )

def get_file_from_key(bucket: str, key: str) -> str:
    try:
        obj = S3_RESOURCE.Object(bucket, key)
        return obj.get()["Body"].read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error retrieving {key} from B2: {e}")
        return None

class B2JinjaInterpreter(ExecutionEnvironment):
    language: str = "jinja"
    
    def __init__(
        self,
        session_id: str,
        mcp_servers: Dict,
        bucket_name: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        prefix: str
    ):
        self.state = {}
        self.b2_config = {
            "bucket_name": bucket_name,
            "endpoint_url": endpoint_url,
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
            "prefix": prefix
        }
        init_s3(endpoint_url, access_key_id, secret_access_key)
        self.file_map = self._build_file_map()
        self.storage = {}
        self.session_id = session_id
        self.static_tools = mcp_servers
        super().__init__(session_id=session_id, mcp_servers=mcp_servers)

    def _build_file_map(self) -> Dict[str, str]:
        file_map = {}
        try:
            for obj in S3_RESOURCE.Bucket(self.b2_config["bucket_name"]).objects.filter(Prefix=self.b2_config["prefix"]):
                base_name = os.path.splitext(os.path.basename(obj.key))[0]
                file_map[base_name] = obj.key
        except Exception as e:
            logger.error(f"Error building file map: {e}")
        return file_map

    def _create_jinja_environment(self) -> Environment:
        env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,
            extensions=['jinja2.ext.do'],
            autoescape=False
        )
        
        # Add static tools as global functions
        env.globals.update(self.static_tools)
        
        # Add B2 file loader function
        def load_b2_file(name: str) -> str:
            if file_key := self.file_map.get(name):
                return get_file_from_key(self.b2_config["bucket_name"], file_key)
            raise InterpreterError(f"File '{name}' not found in B2 storage")
        env.globals['load_b2_file'] = load_b2_file
        
        return env

    def parse_error_logs(self, execution_logs: str) -> str:
        error_lines = []
        for line in execution_logs.split('\n'):
            if any(keyword in line for keyword in ['TemplateSyntaxError', 'TemplateRuntimeError', 'UndefinedError']):
                error_lines.append(line.strip())
        return ' | '.join(error_lines)

    def __call__(self, code_action: str, additional_variables: Dict, return_type: str = "string") -> Tuple[str, str, bool]:
        self.state.update(additional_variables)
        jinja_env = self._create_jinja_environment()
        
        try:
            template = jinja_env.from_string(code_action)
            template.globals.update(self.state)
            template.globals.update(additional_variables)
            template.globals.update(self.static_tools)
            template.globals.update(self.file_map)
            rendered = template.render()
            return rendered, self.state.get("logs", ""), False
        except TemplateSyntaxError as e:
            error_msg = f"Template syntax error: {e.message} (Line {e.lineno})"
            raise InterpreterError(error_msg)
        except TemplateRuntimeError as e:
            error_msg = f"Template runtime error: {str(e)}"
            raise InterpreterError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            raise InterpreterError(error_msg)

    def parse_code_blobs(self, code_blob: str) -> str:
        pattern = r"```(?:jinja)?\n(.*?)\n```"
        matches = re.findall(pattern, code_blob, re.DOTALL)
        if not matches:
            try:
                Template(code_blob)  # Validate basic syntax
                return code_blob
            except TemplateSyntaxError as e:
                error_msg = self.parse_error_logs(str(e))
                raise ValueError(f"Invalid Jinja template:\n{error_msg}\n\nEnclose templates in ```jinja...```")
        return "\n".join(match.strip() for match in matches)

    def get_storage_id(self, next_step_id: int) -> str:
        return f"{self.b2_config['prefix']}/session/{self.session_id}/step_{next_step_id}.txt"

    def set_storage(self, next_step_id: int, code_action: str, observations: List[Dict[str, Any]] = None):
        temp_key = self.get_storage_id(next_step_id)
        content = observations[0]['content'] if observations else self(code_action, {})[0]
        
        try:
            S3_RESOURCE.Bucket(self.b2_config["bucket_name"]).put_object(
                Key=temp_key,
                Body=content.encode('utf-8'),
                ContentType='text/plain'
            )
            self.storage[next_step_id] = temp_key
        except Exception as e:
            raise InterpreterError(f"Failed to store template result: {str(e)}")

    def get_final_storage(self) -> str:
        if not self.storage:
            return ""
        max_step = max(self.storage.keys())
        return self._retrieve_from_storage(max_step)

    def get_storage(self, next_step_id: int) -> str:
        return self._retrieve_from_storage(next_step_id)

    def _retrieve_from_storage(self, step_id: int) -> str:
        if step_id not in self.storage:
            return ""
        try:
            return get_file_from_key(self.b2_config["bucket_name"], self.storage[step_id])
        except Exception as e:
            logger.error(f"Error retrieving storage {step_id}: {e}")
            return ""

    def reset_storage(self):
        for key in self.storage.values():
            try:
                S3_RESOURCE.Object(self.b2_config["bucket_name"], key).delete()
            except Exception as e:
                logger.error(f"Error cleaning storage: {e}")
        self.storage = {}