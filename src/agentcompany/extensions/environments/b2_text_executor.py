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


class QuestionAnswer(BaseModel):
    question: str
    answer: Optional[str]
    success: bool
    
    
def get_exa_web_text(prompt: str) -> str:
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
        messages=[
            {"role":"user","content": prompt}    
        ],
    )
    response = completion.choices[0].message.content
    return response
    

def answer_from_data(data: str, question: str) -> QuestionAnswer:
    """
    Generate a JSON array for the user text using the Gemini API.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = QuestionAnswer(question=question, answer=None, success=False)
    # TODO fix the prompt by switching to flow agent here
    prompt = f"""
    You have to answer the question with a plain text value and not formatting based on the below data:
    
    {data} \n\n
    
    Question:
    {question}
    
    You can extrapolate the answer if sufficient information is not provided but you can derive the answer from the data.
    """
    # Generate Answer
    completion = client.chat.completions.create(
        model="o4-mini",
        messages=[{"role":"user","content": prompt}],
    )
    code_action = completion.choices[0].message.content
    print(f"AnswerFromData -> Prompt: {prompt} Code Action: {code_action}")
    # Judge Answer
    judge_prompt = f"""
    Question: {question}
    Answer: {code_action}
    Is any answer given in the answer? Answer with True or False.
    """.strip()
    print(f"Judge Prompt: {judge_prompt}")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": judge_prompt}],
    )
    judge = completion.choices[0].message.content
    print(f"Judge: {judge}")
    if judge.lower() == "true":
        response.success = True
        response.answer = code_action
    return response


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
        super().__init__(session_id=session_id, mcp_servers=mcp_servers)
    
    def parse_code_blob(self, code_blob: str) -> Template:
        code_blob = code_blob.strip()
        t = Template(code_blob)
        if not t.is_valid():
            raise InterpreterError("Invalid code blobs for Python template string.")
        return code_blob
    
    
    def setup_index(self) -> None:
        files = list_files(self.b2_config["bucket_name"], self.b2_config["prefix"])
        for file_key in files:
            if file_key.endswith("content.txt"):
                task_file = file_key.replace("content.txt", "task.txt")
                if not check_key_exists(self.b2_config["bucket_name"], task_file):
                    print(f"File {task_file} does not exist.")
                    continue
                task_text = get_text_from_key(self.b2_config["bucket_name"], task_file)
                file_text = get_text_from_key(self.b2_config["bucket_name"], file_key)
                self.save_in_long_term_memory(file_key, task_text, file_text)

    def get_b2_file_data(self, code_action: str) -> Optional[str]:
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
    
    
    def get_file_data(self, code_action: str) -> Optional[str]:
        task_index = self.pinecone_client.Index("b2_text_interpreter/task")
        # Search the dense index
        # TODO add reranking
        results = task_index.search(
            namespace=self.b2_config["prefix"],
            query={
                "top_k": 1,
                "inputs": {
                    'text': code_action
                }
            }
        )
        hits = results["results"]["hits"]
        print(f"Hits: {hits}")
        raise NotImplementedError("Pinecone search is not implemented yet.")
    
    def save_in_long_term_memory(self, file_key: str, code_action: str, answer: str) -> None:
        task_index = self.pinecone_client.Index("b2_text_interpreter/task")
        task_index.upsert_records(
            self.b2_config["prefix"],
            [
                {
                    "_id": file_key,
                    "text": code_action,
                }
            ]
        ) 
        answer_index = self.pinecone_client.Index("b2_text_interpreter/answer")
        answer_index.upsert_records(
            self.b2_config["prefix"],
            [
                {
                    "_id": file_key,
                    "text": answer,
                }
            ]
        )      
        
    def save_data(self, code_action: str, answer: str) -> None:
        random_uuid = randint(0, 1000000)
        file_key = f"{self.b2_config['prefix']}/session/{self.session_id}/{random_uuid}.content.txt"
        task_key = file_key.replace(f"{random_uuid}.content.txt", f"{random_uuid}.task.txt")
        store_file(self.b2_config["bucket_name"], file_key, answer.encode("utf-8"))
        store_file(self.b2_config["bucket_name"], task_key, code_action.encode("utf-8"))    
        self.save_in_long_term_memory(file_key, code_action, answer)
        
        
    def web_call(self, code_action: str) -> Optional[str]:
        # Get Files from Memory
        response: QuestionAnswer = QuestionAnswer(question=code_action, answer=None, success=False)
        # Look in memory
        file_data = self.get_file_data(code_action)
        if file_data is not None:
            response = answer_from_data(file_data, code_action)
        # Look in EXA Web
        if not response.success:
            exa_answer = get_exa_web_text(code_action)
            if not exa_answer.startswith("I am sorry"):
                response.answer = exa_answer
                response.success = True
                self.save_data(code_action, exa_answer)
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
                text = get_url_as_text(url)
                data = f"{file_data}\n\n" +  "-" * 80  + f"{url}\n\n{text}"
                response = answer_from_data(data, code_action)
                if response.success:
                    self.save_data(code_action, response.answer)
        
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