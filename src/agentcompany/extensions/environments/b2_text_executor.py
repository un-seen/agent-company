import os
import pinecone
import re
import shlex
from random import randint
import logging
from string import Template 
from typing import Any, Dict, List, Optional, Tuple, Callable
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from agentcompany.driver.markdown import list_of_dict_to_markdown_table
from agentcompany.extensions.environments.exceptions import InterpreterError
from agentcompany.extensions.environments.base import ExecutionEnvironment
from agentcompany.mcp.base import ModelContextProtocolImpl
from typing import Optional, Set
from pydantic import BaseModel


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

def check_key_exists(bucket_name: str, key: str) -> bool:
    """
    Return True if the given key exists in the specified S3 bucket,
    False if it does not exist. Raises on other errors (e.g., permissions).
    """
    global S3_RESOURCE
    obj = S3_RESOURCE.Object(bucket_name, key)
    try:
        # .load() will issue a HEAD request under the hood
        obj.load()
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchKey"):
            return False
        # re-raise other unexpected exceptions (e.g., 403 Forbidden)
        raise
    return True    


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
        print(f"Error storing file {key}: {e}")

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

def collapse_dollar_runs(text: str) -> str:
    """
    Replace every occurrence of two or more consecutive '$' characters with a single '$'.

    Args:
        text: The input string potentially containing runs of '$'.

    Returns:
        A new string where any '$$...' sequence is collapsed to '$'.
    """
    # \${2,} matches any run of 2 or more '$'
    return re.sub(r"\${2,}", "$", text)



class B2TextInterpreter(ExecutionEnvironment):
    
    language: str = "string"
    
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
        self.pinecone_client = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.vector_index = self.pinecone_client.Index("b2-interpreter")
        super().__init__(session_id=session_id, mcp_servers=mcp_servers)
    
    def parse_code_blob(self, code_blob: str) -> Template:
        code_blob = code_blob.strip()
        t = Template(code_blob)
        if not t.is_valid():
            raise InterpreterError("Invalid code blobs for Python template string.")
        return code_blob
    
    
    def setup_vector_index(self) -> None:
        files = list_files(self.b2_config["bucket_name"], self.b2_config["prefix"])
        for file_key in files:
            if file_key.endswith("content.txt"):
                task_file = file_key.replace("content.txt", "task.txt")
                if not check_key_exists(self.b2_config["bucket_name"], task_file):
                    print(f"File {task_file} does not exist.")
                    continue
                task_text = get_text_from_key(self.b2_config["bucket_name"], task_file)
                file_text = get_text_from_key(self.b2_config["bucket_name"], file_key)
                namespace = self.get_vector_namespace("task")
                self.vector_index.upsert_records(
                    namespace,
                    [
                        {
                            "_id": file_key,
                            "text": f"Task: {task_text} \n \n Answer: {file_text}",
                        }
                    ]
                ) 

    def get_b2_file_data(self, code_action: str) -> Optional[str]:
        from agentcompany.extensions.environments.web_executor import quick_word_match
        # Get Files from Memory
        files = list_files(self.b2_config["bucket_name"], self.b2_config["prefix"])
        data = None
        # Default to EXA Web if no files found
        if len(files) > 0:
            # Create Context
            data = ""
            for index, file_key in enumerate(files, 1):
                if file_key.endswith("content.txt"):
                    task_file = file_key.replace("content.txt", "task.txt")
                    if not check_key_exists(self.b2_config["bucket_name"], task_file):
                        print(f"File {task_file} does not exist.")
                        continue
                    task_text = get_text_from_key(self.b2_config["bucket_name"], task_file)
                    file_text = get_text_from_key(self.b2_config["bucket_name"], file_key)
                    if len(task_text) > 0 and len(file_text) > 0 and quick_word_match(task_text, code_action, case_insensitive=True):
                        data += f"File: {file_key}\n"
                        data += f"Task: {task_text}\n"
                        data += f"Content: {file_text}\n"
                        data += "-" * 80 + "\n"
        return data
    
    def get_vector_namespace(self, _type: str) -> str:
        return self.b2_config["prefix"].replace("/", "_") + "_" + _type
    
    def file_qa(self, code_action: str, count: int) -> Optional[str]:
        namespace = self.get_vector_namespace("task")
        # TODO add reranking
        task_results = self.vector_index.search(
            namespace=namespace,
            query={
                "top_k": count,
                "inputs": {
                    'text': code_action
                }
            }
        )
        hits = task_results["result"]["hits"]
        ids = [hit["_id"] for hit in hits]
        task_list = [hit["fields"]["text"] for hit in hits]
        data = ""
        for file_key, task_answer_text in zip(ids, task_list):
            data += f"File: {file_key}\n"
            data += f"{task_answer_text}\n"
            data += "-" * 80 + "\n"

        return data if len(data) > 0 else None
        
    def save_qa(self, code_action: str, answer: str) -> None:
        random_uuid = randint(0, 1000000)
        file_key = f"{self.b2_config['prefix']}/session/{self.session_id}/{random_uuid}.content.txt"
        task_key = file_key.replace(f"{random_uuid}.content.txt", f"{random_uuid}.task.txt")
        store_file(self.b2_config["bucket_name"], file_key, answer.encode("utf-8"))
        store_file(self.b2_config["bucket_name"], task_key, code_action.encode("utf-8"))    
        # Vector
        namespace = self.get_vector_namespace("task")
        self.vector_index.upsert_records(
            namespace,
            [
                {
                    "_id": file_key,
                    "text": f"Task: {code_action} \n \n Answer: {answer}",
                }
            ]
        ) 
        
    def web_qa(self, code_action: str) -> Optional[str]:
        # Get Files from Memory
        from agentcompany.extensions.environments.web_executor import exa_web_qa, QuestionAnswer, answer_from_data
        response: QuestionAnswer = QuestionAnswer(question=code_action, answer=None, success=False)
        # Look in memory
        file_data = self.file_qa(code_action, count=3)
        if file_data is not None:
            response = answer_from_data(file_data, code_action)
        # Look in EXA Web
        if not response.success:
            exa_answer = exa_web_qa(code_action)
            if not exa_answer.startswith("I am sorry"):
                response.answer = exa_answer
                response.success = True
                self.save_qa(code_action, exa_answer)
            else:
                response.answer = exa_answer
                response.success = False
        # Look in Web with Reasoning
        if not response.success:
            from agentcompany.extensions.tools.brave import brave_web_search
            from agentcompany.extensions.tools.jina import get_url_as_text
            search_urls = brave_web_search(code_action)
            if len(search_urls) > 0:
                url = search_urls[0]["url"]
                max_attempt = 3
                attempt = 0
                while True:
                    try:
                        text = get_url_as_text(url)
                        break
                    except Exception as e:
                        if attempt >= max_attempt:
                            return response.answer
                        print(f"Error getting URL: {url} - {e}")
                        attempt += 1
                data = f"{file_data}\n\n" +  "-" * 80  + f"{url}\n\n{text}"
                response = answer_from_data(data, code_action)
                if response.success:
                    self.save_qa(code_action, response.answer)
        
        return response.answer
    
    def get_identifiers(self, code_action: str) -> List[str]:
        """
        Get identifiers from the code action.
        """
        code_action = collapse_dollar_runs(code_action)
        template = Template(code_action)
        identifiers = template.get_identifiers()
        return identifiers
    
    def __call__(self, code_action: str, additional_variables: Dict, return_type: str = "string") -> Tuple[str, str, bool]:
        code_action = collapse_dollar_runs(code_action)
        self.state.update(additional_variables)
        template = Template(code_action)
        return template.substitute(self.state), "", False        
        
    def attach_variables(self, variables: Dict):
        self.state.update(variables)

    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        self.static_tools.update(mcp_servers)

    def parse_function(self, code_blob):
        raise NotImplementedError("Function parsing is not implemented in this environment.")
    
    def parse_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse context to extract relevant information"""
        return context
    
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