import requests
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Callable
from PIL import Image
import numpy as np
import io
import re
import json
import pandas as pd
from agentcompany.extensions.environments.base import ExecutionEnvironment
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.driver.markdown import json_to_markdown
import logging

DEFAULT_MAX_LEN_OUTPUT = 5000


logger = logging.getLogger(__name__)

def fix_final_answer_code(code: str) -> str:
    """
    Sometimes an LLM can try to assign a variable to final_answer, which would break the final_answer() tool.
    This function fixes this behaviour by replacing variable assignments to final_answer with final_answer_variable,
    while preserving function calls to final_answer().
    """
    # First, find if there's a direct assignment to final_answer
    # Use word boundary and negative lookbehind to ensure it's not an object attribute
    assignment_pattern = r"(?<!\.)(?<!\w)\bfinal_answer\s*="
    if "final_answer(" not in code.lower() or not re.search(assignment_pattern, code.lower()):
        # If final_answer tool is not called in this blob, then doing the replacement is hazardous because it could false the model's memory for next steps.
        # Let's not modify the code and leave the subsequent assignment error happen.
        return code

    # Pattern for replacing variable assignments
    # Looks for 'final_answer' followed by '=' with optional whitespace
    # Negative lookbehind ensures we don't match object attributes
    assignment_regex = r"(?<!\.)(?<!\w)(\bfinal_answer)(\s*=)"
    code = re.sub(assignment_regex, r"final_answer_variable\2", code)

    # Pattern for replacing variable usage but not function calls
    # Negative lookahead (?!\s*\() ensures we don't match function calls
    # Negative lookbehind (?<!\.|\w) ensures we don't match object methods or other variables
    variable_regex = r"(?<!\.)(?<!\w)(\bfinal_answer\b)(?!\s*\()"
    code = re.sub(variable_regex, "final_answer_variable", code)
    return code

class TfServingInput(TypedDict):
    endpoint: str
    file_url: str    


class LocalTfServingInterpreter(ExecutionEnvironment):
    """
    A simple executor class for TensorFlow Serving that takes as input:
      - A dictionary of TFServing endpoints (model_name: endpoint_url).
      - An action dictionary with 'model_name' and 'file_url' to process.

    The executor fetches the file from the URL, preprocesses it (placeholder logic),
    and sends it to the specified TFServing endpoint via REST API.

    **Warning:** This is a simplified proof-of-concept. Real-world use requires
    proper file validation, preprocessing, and error handling.
    """
    language: str = "json"
    
    def __init__(
        self,
        session_id: str,
        mcp_servers: Dict[str, ModelContextProtocolImpl],
        endpoints: List[Dict[str, str]],
        allowed_endpoints: Optional[List[str]] = None,
    ):
        # Dictionary of TFServing endpoints (model_name: endpoint_url).
        self.static_tools = mcp_servers
        self.endpoints = endpoints if endpoints is not None else []
        self.endpoint_schema = {}
        for endpoint in self.endpoints:
            self.endpoint_schema[endpoint["model_name"]] = endpoint["model_url"]
        # Define a whitelist of allowed model names.
        if allowed_endpoints is None:
            self.allowed_endpoints = list(self.endpoint_schema.keys()) if self.endpoint_schema else []
        else:
            self.allowed_models = allowed_endpoints

        self.state: Dict[str, Any] = {}  # Additional state variables (if needed)
        self.max_print_outputs_length = DEFAULT_MAX_LEN_OUTPUT
        self.print_outputs = ""  # Log buffer for all command outputs
        super().__init__(session_id, mcp_servers)
        
    def attach_variables(self, variables: dict):
        self.state.update(variables)

    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        self.static_tools.update(mcp_servers)
    
    def parse_function(self, code_blob: str) -> Dict[str, Callable]:
        raise NotImplementedError("parse_function not implemented")
    
    def _is_model_allowed(self, model_name: str) -> bool:
        """
        Check if the specified model name is in the whitelist.
        """
        return model_name in self.allowed_models

    def _fetch_file(self, file_url: str) -> bytes:
        """
        Fetch the file content from the given URL.
        """
        try:
            response = requests.get(file_url, timeout=10)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch file from {file_url}: {str(e)}")

    def _preprocess_file(self, file_name: str, file_content: bytes, model_name: str) -> List[np.ndarray]:
        """
        Placeholder for preprocessing the file content based on the model.
        Replace with actual logic (e.g., image resizing, text tokenization).
        """
        # Load image from bytes
        try:
            image = Image.open(io.BytesIO(file_content)).convert('RGB')
            # Resize to a common input size (e.g., 224x224 for many image models)
            image = image.resize((512, 512))
            # Convert to numpy array and normalize to [0, 1]
            image_array = np.array(image) / 255.0
            # Convert to list of lists (tensor shape: [224, 224, 3] for RGB)
            tensor = image_array.tolist()
            # Return as a list of one tensor (batch size 1)
            return [tensor]
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def __call__(self, code_action: str, additional_variables: Dict, return_type: str = "string") -> Tuple[str, str, bool]:
        """
        Executes the TFServing prediction request based on the provided action.

        Args:
            action (Dict[str, Any]): A dictionary with 'model_name' and 'file_url'.
            additional_variables (Dict[str, Any], optional): Additional environment variables
                (not used in this executor but kept fofr compatibility).

        Returns:
            Tuple[str, str, bool]:
              - The prediction output from TFServing (as a JSON string),
              - The accumulated log of outputs,
              - A boolean flag is_final_answer (always True for this executor).
        """
        # Extract model_name and file_url from the action dictionary
        try:
            code_action = json.loads(code_action)
            model_name = code_action["model_name"]
            file_url = code_action["file_url"]
        except KeyError as e:
            raise ValueError(f"Action dictionary missing required key: {str(e)}")

        # Validate the model name
        if not self._is_model_allowed(model_name):
            raise ValueError(f"Model '{model_name}' is not allowed. Allowed models: {self.allowed_models}")

        # Get the corresponding endpoint
        endpoint = self.endpoint_schema.get(model_name)
        if not endpoint:
            raise ValueError(f"No endpoint defined for model '{model_name}'")

        # Fetch and preprocess the file
        file_content = self._fetch_file(file_url)
        input_data = self._preprocess_file(file_url, file_content, model_name)

        # Prepare the TFServing payload
        payload = {"instances": input_data}

        # Execute the TFServing request
        try:
            response = requests.post(endpoint, json=payload, timeout=240)
            response.raise_for_status()
            output = response.json()  # Return raw JSON string
            markdown_table = json_to_markdown(output)
            # Append to log buffer (truncate if necessary)
            self.print_outputs += str(output) + "\n"
            if len(self.print_outputs) > self.max_print_outputs_length:
                self.print_outputs = self.print_outputs[-self.max_print_outputs_length:]
            return markdown_table, self.print_outputs, False
        except requests.exceptions.RequestException as e:
            raise ValueError(f"TFServing request failed for {endpoint}: {str(e)}")        

    def set_storage(self, next_step_id: int, code_action: str):
        # No memory available for TFServing
        logger.info("set_storage called but no storage available in TfServing")
        return None
    
    def get_storage_id(self, next_step_id: int) -> str:
        # No memory available for TFServing
        logger.info("get_storage_id called but no storage available in TfServing")
        return ""
    
    def get_storage(self, storage_id):
        logger.info("get_storage called but no memory available in TfServing")
        return {}
    
    def get_final_storage(self) -> pd.DataFrame:
        logger.info("get_final_storage called but no storage available in TfServing")
        return None
    
    def parse_code_blobs(self, code_blob: str) -> str:
        """Parses the LLM's output to get any code blob inside. Will return the code directly if it's code."""
        pattern = r"```(?:json)?\n(.*?)\n```"
        matches = re.findall(pattern, code_blob, re.DOTALL)
        if len(matches) == 0:
            try:  # Maybe the LLM outputted a code blob directly
                json.loads(code_blob)
                return code_blob
            except SyntaxError:
                pass
            if "final" in code_blob.lower() and "answer" in code_blob.lower():
                raise ValueError(f"""
                        Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
                        Here is your code snippet:
                        {code_blob}
                        It seems like you're trying to return the final answer, you can do it as follows:
                        Code:
                        ```json
                        {{
                            "final_answer": "Your final answer here"
                        }}
                        ```<end_code>""".strip())
            raise ValueError(f"""Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
                    Here is your code snippet:
                    {code_blob}
                    Make sure to include code with the correct pattern, for instance:
                    Thoughts: Your thoughts
                    Code:
                    ```json
                    {{
                       /* Your JSON code here */
                    }}
                    ```<end_code>""".strip())
        return fix_final_answer_code("\n\n".join(match.strip() for match in matches))