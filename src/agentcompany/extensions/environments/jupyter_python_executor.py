import os
import ast
import re
import pickle
import time
from queue import Empty
import base64
import nbformat
from nbformat import v4 as nb_v4
from jupyter_client import KernelManager
from nbformat.notebooknode import NotebookNode
from typing import Dict, Any, Tuple, Union, List, Callable, Set, Optional
import logging
import pandas as pd

import json
from agentcompany.extensions.environments.base import ExecutionEnvironment

logger = logging.getLogger(__name__)



_IMPORT_PACKAGE_MAP = {
    # Computer Vision
    "cv2": "opencv-python",
    
    # Web Scraping
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    
    # Data Science
    "sklearn": "scikit-learn",
    "pd": "pandas",
    "np": "numpy",
    "plt": "matplotlib",
    
    # Image Processing
    "PIL": "pillow",
    
    "IPython": "ipython",
    
    # Configuration
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    
    # Database
    "psycopg2": "psycopg2-binary",
    
    # Utilities
    "dateutil": "python-dateutil",
    "jwt": "pyjwt",
    "django": "django",
    "flask": "flask"
}

_STANDARD_LIBRARY_MODULES = {
    "sys", "os", "re", "json", "ast", "subprocess", "typing", "logging",
    "importlib", "collections", "datetime", "math", "random", "socket",
    "argparse", "itertools", "functools", "threading", "pathlib", "csv",
    "html", "http", "urllib", "xml", "email", "ssl", "hashlib", "base64",
    "io", "time", "unittest", "pdb", "traceback", "zipfile", "sqlite3",
    "glob", "pickle", "tempfile", "uuid", "webbrowser", "ctypes", "queue",
    "asyncio", "signal", "warnings", "weakref", "dataclasses", "enum",
    "statistics", "pprint", "textwrap", "shutil", "doctest", "profile"
}

def get_pip_package(module_name: str) -> Optional[str]:
    """
    Maps Python import names to their corresponding PyPI package names.
    
    Args:
        module_name (str): The name used in import statements
        
    Returns:
        str: PyPI package name if mapping exists
        None: For standard library modules
        module_name: For unmapped packages (assumed same as import name)
    """
    # Check if it's a known standard library module
    if module_name in _STANDARD_LIBRARY_MODULES:
        return None
        
    # Return mapped package name if exists
    return _IMPORT_PACKAGE_MAP.get(module_name, module_name)

def push_variables_to_kernel(local_vars: dict, kernel_client):
    """
    For each key/value in local_vars, serialize and define it in the Jupyter kernel via code injection.
    """
    for var_name, value in local_vars.items():
        # 1) Serialize with pickle
        pickled_bytes = pickle.dumps(value)
        b64_str = base64.b64encode(pickled_bytes).decode('utf-8')
        
        # 2) Construct code to decode and assign in the kernel
        # We wrap this in triple-quotes to avoid special char issues
        code = f"""
        import pickle, base64
        {var_name} = pickle.loads(base64.b64decode(\"\"\"{b64_str}\"\"\"))
        print(\"Assigned variable '{var_name}' in kernel.\")
        """
        # 3) Execute on the remote kernel
        msg_id = kernel_client.execute(code)
        
        # Optional: wait for the execution result or handle outputs
        # But typically you can just dispatch and keep going
        # To block until complete:
        while True:
            msg = kernel_client.get_iopub_msg(timeout=1)
            if msg['msg_type'] == 'stream':
                print(msg['content']['text'], end='')
            if (msg['msg_type'] == 'status' and
                msg['content']['execution_state'] == 'idle'):
                break
            
class JupyterPythonInterpreter(ExecutionEnvironment):
    
    language: str = "python"
    
    def __init__(
        self,
        session_id: str,
        mcp_servers: Dict,
        additional_authorized_imports: List[str],
        notebook_path: str = "/tmp/jupyter_notebooks",
    ):
        self.session_id = session_id
        self.notebook_path = notebook_path
        self.notebook_name = f"execution_{session_id}.ipynb"
        self.cell_counter = 0
        
        # Create notebook directory if needed
        os.makedirs(self.notebook_path, exist_ok=True)
        
        # Initialize kernel
        self.km: KernelManager = KernelManager()
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        
        # Create new notebook structure
        
        self.notebook: NotebookNode = nb_v4.new_notebook()
        self._add_initial_cells()
        
        super().__init__(session_id, mcp_servers)
        
    def _add_initial_cells(self):
        """Add initial setup cells to the notebook"""
        init_code = [
            "# Initialization cell",
            "import warnings",
            "warnings.filterwarnings('ignore')"
        ]
        self._add_execute_cell("\n".join(init_code))
        self._execute_cell(0)
        
    def __del__(self):
        """Cleanup kernel when instance is destroyed"""
        self.km.shutdown_kernel()
        
    def __call__(
        self, 
        code_action: str,
        additional_variables: dict,
        return_type: str = "string"
    ) -> Tuple[Any, str, bool]:
        # Push variables first
        push_variables_to_kernel(additional_variables, self.kc)
        
        # Add cell with result capture
        wrapped_code = f"""
        import pickle
        import base64

        try:
            # User's original code
            {code_action}
            
            # Capture the last expression result
            __result__ = _
        except Exception as e:
            __result__ = e  # Capture exceptions as results

        # Serialize and clean up
        try:
            __serialized__ = base64.b64encode(pickle.dumps(__result__)).decode('utf-8')
            print(__serialized__)  # This will be our captured output
        except Exception as e:
            print(f"SERIALIZATION_ERROR: {{str(e)}}")
        del __result__  # Clean up namespace
        """
        # Execute wrapped code
        cell_index = self._add_execute_cell(wrapped_code)
        outputs, error_logs = self._execute_cell(cell_index)
        
        # Extract serialized result
        serialized = ""
        for output in outputs:
            if 'text/plain' in output:
                text = output['text/plain']
                if text.strip():
                    serialized = text.strip()
                    break

        # Deserialize the result
        result = None
        errors = []
        if serialized.startswith("SERIALIZATION_ERROR:"):
            errors.append(serialized)
        elif serialized:
            try:
                result = pickle.loads(base64.b64decode(serialized))
            except Exception as e:
                errors.append(f"Deserialization failed: {str(e)}")
        else:
            errors.append("No output captured from kernel")
        
        # Handle exceptions that were captured
        if isinstance(result, Exception):
            errors.append(f"Execution error: {str(result)}")
            result = None
            
        return result, "\n".join(error_logs + errors), bool(error_logs or errors)

    
    def _add_execute_cell(self, code: str) -> int:
        """Add a new code cell to the notebook"""
        new_cell = nb_v4.new_code_cell(source=code)
        self.notebook.cells.append(new_cell)
        self.cell_counter += 1
        return self.cell_counter - 1

    def _extract_markdown_code(self, code_blob: str) -> str:
        """Extracts Python code from markdown code block"""
        match = re.search(r"```python\s*\n(.+?)\n*```", code_blob, re.DOTALL)
        if not match:
            raise ValueError("No valid Python code or markdown code block found")
        return match.group(1)

    def _collect_imports(self, tree: ast.AST) -> Set[str]:
        """Collects root package names from all import statements"""
        packages = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    packages.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                packages.add(node.module.split('.')[0])
        return packages
    
    def _install_missing_modules(self, modules: Set[str]):
        """Installs packages that aren't available in the environment"""
        for module in modules:
            if not self._is_module_installed(module):
                logger.info(f"Installing missing module: {module}")
                pip_package_name = get_pip_package(module)
                logger.info(f"Installing pip package: {pip_package_name}")
                if not self._pip_install(pip_package_name):
                    raise ImportError(f"Failed to install required module: {module}")
                
    def parse_function(self, code_blob: str) -> Dict[str, Callable]:
        """
        Parses the given code blob using Python's ast module and returns a dictionary mapping
        function names to their corresponding callable functions.
        
        If the input is not valid Python code, it first attempts to extract the substring inside
        a markdown code block (```python ... ```).

        Parameters:
            code_blob (str): A string containing Python code or markdown with a python code block.

        Returns:
            Dict[str, Callable]: A dictionary where keys are function names and values are callable functions.
        """
        try:
            # Attempt to parse the code blob directly.
            tree = ast.parse(code_blob)
            code_to_compile = code_blob
        except SyntaxError:
            # If parsing fails, extract the code block from markdown.
            code_to_compile = self._extract_markdown_code(code_blob)
            try:
                tree = ast.parse(code_to_compile)
            except SyntaxError as e:
                raise ValueError(f"Extracted code is not valid Python: {e}")
            
        # Identify and install required modules
        required_modules = self._collect_imports(tree)
        self._install_missing_modules(required_modules)
        # Compile and execute the (valid) code in a temporary namespace.
        namespace = {}
        compiled_code = compile(tree, filename="<ast>", mode="exec")
        exec(compiled_code, namespace)

        # Filter the namespace for functions (ignoring imported modules and other objects).
        functions_dict = {
            name: obj
            for name, obj in namespace.items()
            if callable(obj) and isinstance(obj, type(lambda: None))
        }
        return functions_dict
    
    
    def parse_code_blobs(self, code_blob: str) -> str:
        """Parses the LLM's output to get any code blob inside. Will return the code directly if it's code."""
        pattern = r"```(?:python)?\n(.*?)\n```"
        matches = re.findall(pattern, code_blob, re.DOTALL)
        if len(matches) == 0:
            try:  # Maybe the LLM outputted a code blob directly
                ast.parse(code_blob)
                return code_blob
            except SyntaxError:
                pass
            if "final" in code_blob and "answer" in code_blob:
                raise ValueError(f"""
                        Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
                        Here is your code snippet:
                        {code_blob}
                        It seems like you're trying to return the final answer, you can do it as follows:
                        Code:
                        ```py
                        final_answer("YOUR FINAL ANSWER HERE")
                        ```<end_code>""".strip())
            raise ValueError(f"""Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
                    Here is your code snippet:
                    {code_blob}
                    Make sure to include code with the correct pattern, for instance:
                    Thoughts: Your thoughts
                    Code:
                    ```py
                    # Your python code here
                    ```<end_code>""".strip())
        return "\n\n".join(match.strip() for match in matches)
    
    def _execute_cell(self, cell_index: int) -> Tuple[Any, str]:
        """Execute cell with extended timeouts and output filtering"""
        cell = self.notebook.cells[cell_index]
        msg_id = self.kc.execute(cell.source)
        
        outputs = []
        error_logs = []
        last_activity = time.time()
        while True:
            try:
                # Use dynamic timeout based on last activity
                elapsed = time.time() - last_activity
                timeout = max(300, 60 - elapsed)  # Allow up to 5 minutes between messages
                msg = self.kc.get_iopub_msg(timeout=timeout)
                
                if msg['parent_header'].get('msg_id') == msg_id:
                    msg_type = msg['msg_type']
                    content = msg['content']
                    last_activity = time.time()

                    # Handle different message types
                    if msg_type == 'execute_result':
                        outputs.append(content['data'])
                    elif msg_type == 'stream':
                        # Filter TensorFlow/XLA noise
                        if 'XLA service' not in content['text'] and 'cuDNN version' not in content['text']:
                            outputs.append({'text/plain': content['text']})
                    elif msg_type == 'error':
                        error_logs.extend(content['traceback'])
                    elif msg_type == 'status':
                        if content['execution_state'] == 'idle':
                            break

            except Empty:
                if time.time() - last_activity > 300:  # 5 minute global timeout
                    error_logs.append("Timeout: No kernel activity for 5 minutes")
                    break
                continue
                
            except Exception as e:
                error_logs.append(f"Kernel error: {str(e)}")
                break

        cell.outputs = outputs
        return outputs, "\n".join(error_logs)


    def attach_variables(self, variables: dict):
        """Inject variables into kernel"""
        for name, value in variables.items():
            code = f"{name} = {repr(value)}"
            self.kc.execute(code)

    def attach_mcp_servers(self, mcp_servers: Dict[str, Any]):
        """Inject MCP servers into kernel context"""
        for name, server in mcp_servers.items():
            self.kc.execute(f"{name} = {repr(server)}")

    def parse_error_logs(self, execution_logs: str) -> str:
        """Simplify Jupyter error logs"""
        if 'Traceback (most recent call last)' in execution_logs:
            return execution_logs.split('Traceback')[-1]
        return execution_logs

    def get_storage(self, next_step_id: int) -> str:
        """Get variable from kernel namespace"""
        code = f"print({self.get_storage_id(next_step_id)})"
        output, _ = self._execute_code_snippet(code)
        return output

    def _execute_code_snippet(self, code: str) -> Tuple[str, str]:
        """Execute code snippet directly in kernel"""
        msg_id = self.kc.execute(code)
        outputs = []
        errors = []
        
        while True:
            msg = self.kc.get_iopub_msg(timeout=10)
            if msg['parent_header'].get('msg_id') == msg_id:
                msg_type = msg['msg_type']
                content = msg['content']
                
                if msg_type == 'execute_result':
                    outputs.append(str(content['data']))
                elif msg_type == 'stream':
                    outputs.append(content['text'])
                elif msg_type == 'error':
                    errors.extend(content['traceback'])
                elif msg_type == 'status' and content['execution_state'] == 'idle':
                    break
        
        return "\n".join(outputs), "\n".join(errors)

__all__ = ["JupyterPythonInterpreter"]