import os
import re
import shlex
import logging
from string import Template 
from typing import Any, Dict, List, Optional, Tuple, Callable
import boto3
from botocore.client import Config
from agentcompany.driver.markdown import list_of_dict_to_markdown_table
from agentcompany.extensions.environments.exceptions import InterpreterError
from agentcompany.extensions.environments.base import ExecutionEnvironment
from agentcompany.mcp.base import ModelContextProtocolImpl
from typing import Optional, Set


logger = logging.getLogger(__name__)

S3_RESOURCE = None
# Precompile once at import-time
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

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

def escape_template_dollars(text: str) -> str:
    """
    Escape all dollar signs for use with string.Template:
      - Literal $ → $$
      - $${var} → $$${var}  (so $$ is literal $, then ${var} is a real placeholder)
    
    Placeholders of the form $name or ${name} are left intact.
    """
    # 1) Escape every $ *not* followed by $, '{', letter or underscore
    text = re.sub(r"\$(?![\${A-Za-z_])", r"$$", text)
    # 2) Fix up any literal $$ right before a {…} so it becomes $$${…}
    text = re.sub(r"\$\$\{", r"$$${", text)
    return text


def quick_word_match(s1: str, s2: str, 
                     *, 
                     case_insensitive: bool = True,
                     word_pattern: Optional[re.Pattern] = None
                    ) -> bool:
    """
    Return True if any “word” in s1 also appears in s2.
    
    - Runs in O(len(s1) + len(s2)) time.
    - Uses a set for O(1) lookups.
    - Short‑circuits on first match for s2.
    
    Params:
      s1, s2            : input strings
      case_insensitive  : if True, lower‑cases before matching
      word_pattern      : custom regex for “words” (default = \\b\\w+\\b)
    """
    pat = word_pattern or _WORD_RE
    
    if case_insensitive:
        s1 = s1.lower()
        s2 = s2.lower()
    
    # build set of words from s1
    words1: Set[str] = {m.group() for m in pat.finditer(s1)}
    if not words1:
        return False
    
    # scan s2, stop at first hit
    for m in pat.finditer(s2):
        if m.group() in words1:
            return True
    return False


def get_web_text(prompt: str) -> str:
    """
    Generate a JSON array for the user text using the Gemini API.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.exa.ai",
        api_key=os.environ["EXA_API_KEY"],
    )

    completion = client.chat.completions.create(
        model="exa",
        messages=[{"role":"user","content": prompt}],
    )
    response = completion.choices[0].message.content
    return response

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
        self.state = {}
        self.storage = {}
        self.session_id = session_id
        self.static_tools = mcp_servers
        super().__init__(session_id=session_id, mcp_servers=mcp_servers)
    
    def parse_code_blob(self, code_blob: str) -> Template:
        code_blob = code_blob.strip()
        t = Template(code_blob)
        if t.is_valid():
            return code_blob
        raise InterpreterError("Invalid code blobs for Python template string.")

    
    def get_file_text(self, code_action: str) -> str:
        files = list_files(self.b2_config["bucket_name"], self.b2_config["prefix"])
        content = {}
        for index, file in enumerate(files, 1):
            if file.endswith("content.txt"):
                task_file = file.replace("content.txt", "task.txt")
                if not check_key_exists(self.b2_config["bucket_name"], task_file):
                    logger.warning(f"File {task_file} does not exist.")
                    continue
                task_text = get_text_from_key(self.b2_config["bucket_name"], task_file)
                if quick_word_match(task_text, code_action, case_insensitive=True):
                    content[file] = task_text
                    
        response = []
        for file, task in content.items():
            file_content = get_text_from_key(self.b2_config["bucket_name"], file)
            response.append({"task": task, "content": file_content})
        
        return list_of_dict_to_markdown_table(response)
    
    def get_identifiers(self, code_action: str) -> List[str]:
        """
        Get identifiers from the code action.
        """
        code_action = escape_template_dollars(code_action)
        template = Template(code_action)
        identifiers = template.get_identifiers()
        return identifiers
    
    def text_search(self, code_action: str) -> str:
        web_data = get_web_text(code_action)
        file_data = self.get_file_text(code_action)
        random_uuid = os.urandom(16).hex
        file_key = f"{self.b2_config['prefix']}/session/{self.session_id}/{random_uuid}.content.txt"
        task_key = file_key.replace(f"{random_uuid}.content.txt", f"{random_uuid}.task.txt")
        store_file(self.b2_config["bucket_name"], file_key, web_data.encode("utf-8"))
        store_file(self.b2_config["bucket_name"], task_key, code_action.encode("utf-8"))
        return f"{file_data}\n\n{web_data}"
    
    def __call__(self, code_action: str, additional_variables: Dict, return_type: str = "string") -> Tuple[str, str, bool]:
        code_action = escape_template_dollars(code_action)
        self.state.update(additional_variables)
        template = Template(code_action)
        return template.substitute(self.state), "", False        
        
    def attach_variables(self, variables: Dict):
        self.state.update(variables)

    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        self.static_tools.update(mcp_servers)

    def parse_function(self, code_blob):
        raise NotImplementedError("Function parsing is not implemented in this environment.")
    
    def get_storage_id(self, next_step_id: int) -> str:
        return f"{self.b2_config['prefix']}/session/{self.session_id}/step_{next_step_id}.txt"

    def set_storage(self, next_step_id: int, code_action: str, observations: str = None):
        temp_key = self.get_storage_id(next_step_id)
        if observations:
            store_file(self.b2_config["bucket_name"], temp_key, observations.encode("utf-8"))
        else:
            text = self.__call__(code_action, self.state)[0]
            store_file(self.b2_config["bucket_name"], temp_key, text.encode("utf-8"))
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