import os
import re
import shlex
import logging
import yaml
from typing import Any, Dict, List, Optional, Tuple, Callable
import boto3
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

def check_key_exists(bucket: str, key: str) -> bool:
    global S3_RESOURCE
    try:
        return S3_RESOURCE.Bucket(bucket).Object(key).content_length > 0
    except Exception:
        return False

def delete_file(bucket, key):
    global S3_RESOURCE
    try:
        S3_RESOURCE.Bucket(bucket).Object(key).delete()
        logger.info(f"Deleted {key} from s3://{bucket}")
    except Exception as e:
        logger.error(f"Error deleting {key} from s3://{bucket}: {e}")

def store_file(bucket: str, key: str, file_content: bytes):
    global S3_RESOURCE
    try:
        S3_RESOURCE.Bucket(bucket).put_object(Key=key, Body=file_content)
    except Exception as e:
        logger.error(f"Error storing file {key}: {e}")

def list_files(bucket: str, path: str) -> List[str]:
    global S3_RESOURCE
    try:
        return [obj.key for obj in S3_RESOURCE.Bucket(bucket).objects.filter(Prefix=path)]
    except Exception as e:
        logger.error(f"Error listing files in {bucket}/{path}: {e}")
        return []

def get_text_from_key(bucket: str, key: str) -> str:
    global S3_RESOURCE
    try:
        obj = S3_RESOURCE.Object(bucket, key)
        return obj.get()["Body"].read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error retrieving {key} from B2: {e}")
        return None

def parse_function_call(call_str: str) -> Tuple[Optional[str], Optional[List[str]]]:
    call_str = call_str.strip()
    pattern = r'^([a-zA-Z_]\w*)\((.*)\)$'
    match = re.match(pattern, call_str)
    if not match:
        return None, None
    func_name = match.group(1).lower()
    args_str = match.group(2).strip()
    
    if not args_str:
        return func_name, []
    
    lex = shlex.shlex(args_str, posix=False)
    lex.whitespace_split = True
    lex.whitespace = ','
    args = []
    try:
        for token in lex:
            if token == ',':
                continue
            args.append(token.strip())
    except ValueError:
        return None, None
    
    return func_name, args

def evaluate_yaml(
    code_action: str,
    state: Dict,
    static_tools: Dict[str, Callable],
    file_map: Dict[str, str],
    bucket_name: str,
) -> str:
    try:
        parsed_yaml = yaml.safe_load(code_action)
    except yaml.YAMLError as e:
        error_msg = f"YAML parsing error: {str(e)}"
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            error_msg += f" at line {mark.line + 1}, column {mark.column + 1}"
        raise InterpreterError(error_msg)

    def process_node(node):
        if isinstance(node, dict):
            processed = {}
            for k, v in node.items():
                processed_key = process_node(k)
                processed_value = process_node(v)
                processed[processed_key] = processed_value
            return processed
        elif isinstance(node, list):
            return [process_node(item) for item in node]
        elif isinstance(node, str):
            func_name, args = parse_function_call(node)
            if func_name:
                if func_name not in static_tools:
                    raise InterpreterError(f"Function '{func_name}' not found")
                resolved_args = []
                for arg_str in args:
                    try:
                        arg = yaml.safe_load(arg_str)
                    except yaml.YAMLError as e:
                        raise InterpreterError(f"Invalid YAML in argument '{arg_str}': {e}")
                    resolved_arg = process_node(arg)
                    resolved_args.append(resolved_arg)
                try:
                    result = static_tools[func_name](*resolved_args)
                except Exception as e:
                    raise InterpreterError(f"Error executing '{func_name}': {str(e)}")
                return result
            else:
                if node in state:
                    return state[node]
                base_name = os.path.splitext(node)[0]
                if base_name in file_map:
                    file_key = file_map[base_name]
                    content_str = get_text_from_key(bucket_name, file_key)
                    if content_str is None:
                        raise InterpreterError(f"File '{file_key}' not found")
                    try:
                        content = yaml.safe_load(content_str)
                    except yaml.YAMLError as e:
                        raise InterpreterError(f"Error parsing '{file_key}': {e}")
                    return process_node(content)
                else:
                    return node
        else:
            return node

    try:
        processed_yaml = process_node(parsed_yaml)
    except InterpreterError as e:
        raise e
    except Exception as e:
        raise InterpreterError(f"Processing error: {str(e)}")

    try:
        return yaml.dump(processed_yaml, default_flow_style=False)
    except yaml.YAMLError as e:
        raise InterpreterError(f"YAML serialization error: {str(e)}")

class B2TextInterpreter(ExecutionEnvironment):
    
    language: str = "python_template_string"
    
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
        
        self.b2_config = {
            "bucket_name": bucket_name,
            "endpoint_url": endpoint_url,
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
            "prefix": prefix
        }
        init_s3(endpoint_url, access_key_id, secret_access_key)
        files = [f for f in list_files(bucket_name, prefix) if f.endswith(".txt")]
        self.state = { "files": files }
        self.storage = {}
        self.session_id = session_id
        self.static_tools = mcp_servers
        super().__init__(session_id=session_id, mcp_servers=mcp_servers)
    
    def parse_code_blobs(self, code_blobs):
        raise NotImplementedError("Code blob parsing is not implemented in this environment.")
    
    def __call__(self, code_action: str, additional_variables: Dict, return_type: str = "string") -> Tuple[str, str, bool]:
        # TODO the code action will have variables with dollar sign prefix
        # list the variables and then find on b2 any related files to the text
        # after that call the gemini api to get data on the text
        # then assign a format to the variable name
        # then using data from file + gemini set the value for the variable name
        self.state.update(additional_variables)
        code_action = self.inject_yaml_file_list(code_action.strip())
        try:
            text = evaluate_yaml(
                code_action=code_action,
                state=self.state,
                static_tools=self.static_tools,
                file_map=self.file_map,
                bucket_name=self.b2_config["bucket_name"]
            )
        except InterpreterError as e:
            raise e
        logs = self.state.get("logs", "")
        return text, logs, False
        
    def attach_variables(self, variables: Dict):
        self.state.update(variables)

    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        self.static_tools.update(mcp_servers)

    def parse_function(self, code_blob):
        raise NotImplementedError("Function parsing is not implemented in this environment.")
    
    def get_storage_id(self, next_step_id: int) -> str:
        return f"{self.b2_config['prefix']}/session/{self.session_id}/step_{next_step_id}.yaml"

    def set_storage(self, next_step_id: int, code_action: str, observations: List[Dict[str, Any]] = None):
        temp_key = self.get_storage_id(next_step_id)
        if observations:
            content = yaml.dump(observations, default_flow_style=False)
            store_file(self.b2_config["bucket_name"], temp_key, content.encode("utf-8"))
        else:
            yaml_text = evaluate_yaml(
                code_action=code_action,
                state=self.state,
                static_tools=self.static_tools,
                file_map=self.file_map,
                bucket_name=self.b2_config["bucket_name"]
            )
            store_file(self.b2_config["bucket_name"], temp_key, yaml_text.encode("utf-8"))
        self.storage[next_step_id] = temp_key

    def reset_storage(self):
        for key in self.storage.values():
            delete_file(self.b2_config["bucket_name"], key)
        self.storage = {}

    def get_final_storage(self) -> str:
        if not self.storage:
            return ""
        max_step = max(self.storage.keys())
        content = get_text_from_key(self.b2_config["bucket_name"], self.storage[max_step])
        return content if content else ""

    def get_storage(self, next_step_id: int) -> str:
        if next_step_id not in self.storage:
            return ""
        content = get_text_from_key(self.b2_config["bucket_name"], self.storage[next_step_id])
        return content if content else ""