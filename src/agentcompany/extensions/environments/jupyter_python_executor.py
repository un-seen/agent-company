import os
import ast
import re
import nbformat
from nbformat import v4 as nb_v4
from jupyter_client import KernelManager
from typing import Dict, Any, Tuple, Union, List, Callable, Set
import logging
import pandas as pd
import json
from agentcompany.extensions.environments.base import ExecutionEnvironment

logger = logging.getLogger(__name__)

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
        self.notebook = nb_v4.new_notebook()
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
        additional_variables: Dict,
        return_type: str = "string"
    ) -> Tuple[Union[List[dict], str], str, bool]:
        try:
            # Add new cell with code
            cell_index = self._add_execute_cell(code_action)
            
            # Execute the cell
            output, error_logs = self._execute_cell(cell_index)
            
            # Save notebook state
            self._save_notebook()
            
            return self._format_output(output, return_type), error_logs, bool(error_logs)
            
        except Exception as e:
            error_msg = f"Jupyter execution failed: {str(e)}"
            logger.error(error_msg)
            return "", error_msg, True

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
        """Execute a specific notebook cell"""
        cell = self.notebook.cells[cell_index]
        msg_id = self.kc.execute(cell.source)
        
        outputs = []
        error_logs = []
        
        # Wait for execution to complete
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=30)
                if msg['parent_header'].get('msg_id') == msg_id:
                    msg_type = msg['msg_type']
                    content = msg['content']
                    if msg_type == 'execute_result':
                        outputs.append(content['data'])
                    elif msg_type == 'stream':
                        outputs.append({'text/plain': content['text']})
                    elif msg_type == 'error':
                        error_logs.extend([
                            f"{content['ename']}: {content['evalue']}",
                            "\n".join(content['traceback'])
                        ])
                    elif msg_type == 'status' and content['execution_state'] == 'idle':
                        break
                        
            except Exception as e:
                error_logs.append(f"Kernel communication error: {str(e)}")
                break
                
        # Update cell outputs
        cell.outputs = outputs
        return outputs, "\n".join(error_logs)

    def _save_notebook(self):
        """Save current notebook state to file"""
        nb_path = os.path.join(self.notebook_path, self.notebook_name)
        with open(nb_path, 'w') as f:
            nbformat.write(self.notebook, f, version=4)

    def _format_output(self, outputs: List[Dict], return_type: str) -> str:
        """Format execution outputs based on return type"""
        formatted = []
        for output in outputs:
            if 'text/plain' in output:
                formatted.append(output['text/plain'])
            elif 'text/html' in output:
                formatted.append(output['text/html'])
            elif 'application/json' in output:
                formatted.append(json.dumps(output['application/json']))
        
        final_output = "\n".join(formatted)
        
        if return_type == "dataframe":
            return self._output_to_dataframe(final_output)
        return final_output

    def _output_to_dataframe(self, output: str) -> pd.DataFrame:
        """Convert text output to DataFrame"""
        try:
            return pd.read_json(output)
        except ValueError:
            return pd.DataFrame([output])

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

    # Implement other required methods from base class
    # ... (get_storage_id, set_storage, etc)
    
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