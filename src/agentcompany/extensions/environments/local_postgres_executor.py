
import re
import sqlglot
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple
import traceback
import numpy as np
import pandas as pd
import sqlglot.expressions
from agentcompany.mcp.base import ModelContextProtocolImpl
import psycopg2
import logging
from agentcompany.driver.dict import dict_rows_to_markdown_table
from psycopg2.extras import RealDictCursor
from agentcompany.mcp.utils import truncate_content
from agentcompany.extensions.environments.exceptions import InterpreterError, ERRORS
from agentcompany.extensions.environments.base import ExecutionEnvironment

logger = logging.getLogger(__name__)

BASE_BUILTIN_MODULES = [
    "time",
    "unicodedata",
]

PRINT_OUTPUTS, DEFAULT_MAX_LEN_OUTPUT = "", 50000
OPERATIONS_COUNT, MAX_OPERATIONS = 0, 10000000


def get_iterable(obj):
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, "__iter__"):
        return list(obj)
    else:
        raise InterpreterError("Object is not iterable")

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

class FinalAnswerException(Exception):
    def __init__(self, value):
        self.value = value


def evaluate_ast(pg_conn, node, state, static_tools: Dict[str, ModelContextProtocolImpl], custom_tools, authorized_imports):
    # Check if the expression is a sqlglot SELECT statement.
    # (sqlglot returns expressions from the sqlglot.exp module;
    # adjust the type check if needed.)
    if isinstance(node, sqlglot.exp.Select):
        # Convert the AST to a Postgres-compatible SQL string.
        # Check if NODE uses an MCP server. if yes then call the server and replace the node subtree with the result.
        for idx, statement in enumerate(node.expressions):
            if statement.this in static_tools:
                function_name = statement.this.lower()
                function_call = static_tools[function_name]
                function_arguments = [exp.this for exp in statement.expressions]
                function_response = function_call(*function_arguments)
                response_value = sqlglot.expressions.Literal()    
                if function_call.output_type == "string":
                    response_value.args["this"] = function_response
                    response_value.args["is_string"] = True
                node.args["expressions"] = node.expressions[:idx] + [response_value] + node.expressions[idx+1:]
        sql_query = node.sql(dialect="postgres")
        try:
            with pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql_query)
                result = cursor.fetchall()
            return result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error executing SQL: {error_msg}")
            raise InterpreterError(f"Error executing SQL: {error_msg}")
    else:
        raise InterpreterError(f"Unsupported AST node type: {type(node)}")
    
        
def evaluate_sql_code(
    pg_conn,
    code: str,
    static_tools: Optional[Dict[str, Callable]] = None,
    custom_tools: Optional[Dict[str, Callable]] = None,
    state: Optional[Dict[str, Any]] = None,
    authorized_imports: List[str] = BASE_BUILTIN_MODULES,
    max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
):
    """
    Evaluate a sql expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (str): The code to evaluate.
        static_tools (Optional[Dict[str, Callable]]): The functions that may be called during the evaluation.
        custom_tools (Optional[Dict[str, Callable]]): The functions that may be called during the evaluation.
        state (Optional[Dict[str, Any]]): A dictionary mapping variable names to values.
        authorized_imports (List[str]): List of modules that are allowed to be imported.
        max_print_outputs_length (int): Maximum length for the captured print outputs.
    """
    if not (code.lower().startswith("select") or code.lower().startswith("create") or code.lower().startswith("insert") or code.lower().startswith("update") or code.lower().startswith("delete")):
        code = f"SELECT {code}"
    try:
        expression = sqlglot.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Failed to parse code to create ast on line {e.lineno} due to: {type(e).__name__}\n"
            f"{e.text}"
            f"{' ' * (e.offset or 0)}^\n"
            f"Error: {str(e)}"
        )

    if state is None:
        state = {}
    static_tools = static_tools.copy() if static_tools is not None else {}
    custom_tools = custom_tools if custom_tools is not None else {}
    result = None
    global PRINT_OUTPUTS
    PRINT_OUTPUTS = ""
    global OPERATIONS_COUNT
    OPERATIONS_COUNT = 0

    def final_answer(value):
        raise FinalAnswerException(value)

    static_tools["final_answer"] = final_answer

    try:
        for node in expression:
            result = evaluate_ast(pg_conn, node, state, static_tools, custom_tools, authorized_imports)            
        state["print_outputs"] = truncate_content(PRINT_OUTPUTS, max_length=max_print_outputs_length)
        is_final_answer = False
        return result, is_final_answer
    except FinalAnswerException as e:
        state["print_outputs"] = truncate_content(PRINT_OUTPUTS, max_length=max_print_outputs_length)
        is_final_answer = True
        return e.value, is_final_answer
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Code execution failed at node '{node}' in code '{code}' due to exception:\n{error_trace}"
        raise InterpreterError(error_msg)
    
    
class LocalPostgresInterpreter(ExecutionEnvironment):
    def __init__(
        self,
        mcp_servers: Dict,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        additional_authorized_imports: List[str]
    ):
        self.custom_tools = {}
        self.state = {}
        self.max_print_outputs_length = DEFAULT_MAX_LEN_OUTPUT
        self.additional_authorized_imports = additional_authorized_imports
        self.pg_config = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password
        }
        logger.info(f"Connecting to the database {dbname} on {host}:{port}")
        self.pg_conn = psycopg2.connect(**self.pg_config)
        
        self.sql_schema = []
        logger.info("Fetching schema from the database")
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = [row["table_name"] for row in cur.fetchall()]
            for table in tables:
                self.sql_schema.append(f"Table {table} has columns:")
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;",
                    (table,)
                )
                for row in cur.fetchall():
                    self.sql_schema.append(f"- {row['column_name']}, has data type {row['data_type']}")                
        self.sql_schema = "\n".join(self.sql_schema)
        # TODO: assert self.authorized imports are all installed 
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        # Add base trusted tools to list
        self.static_tools = mcp_servers
    
    def reset_connection(self):
        self.pg_conn.close()
        self.pg_conn = psycopg2.connect(**self.pg_config)

    def parse_error_logs(self, execution_logs: str) -> str:
        # Regex pattern to capture the full InterpreterError including multiline messages
        lines = execution_logs.split('\n')
        error_lines = []
        capture = False
        for line in lines:
            if 'psycopg2.errors' in line:
                # Extract the part of the line starting from 'psycopg2.errors'
                start_idx = line.find('psycopg2.errors')
                error_line = line[start_idx:]
                error_lines.append(error_line)
                capture = True
            elif capture:
                # Check if the current line indicates the end of the error message
                if line.startswith('File ') or line.startswith('During handling') or line.strip() == '':
                    capture = False
                else:
                    error_lines.append(line.strip())
        # Join the captured lines into a single string
        if error_lines:
            return ' '.join(error_lines).strip()
        return execution_logs
        
    def __call__(self, code_action: str, additional_variables: Dict) -> Tuple[Any, str, bool]:
        self.state.update(additional_variables)
        self.reset_connection()
        tupled_rows, is_final_answer = evaluate_sql_code(
            self.pg_conn,
            code_action,
            static_tools=self.static_tools,
            custom_tools=self.custom_tools,
            state=self.state,
            authorized_imports=self.authorized_imports,
            max_print_outputs_length=self.max_print_outputs_length,
        )
        if isinstance(tupled_rows, list) and isinstance(tupled_rows[0], dict):
            markdown_table = dict_rows_to_markdown_table(tupled_rows)
            logs = self.state["print_outputs"]
            return markdown_table, logs, is_final_answer
        else:
            logs = self.state["print_outputs"]
            return "\n".join(tupled_rows), logs, is_final_answer
        
        
    def attach_variables(self, variables: dict):
        self.state.update(variables)

    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        self.static_tools.update(mcp_servers)

    def parse_code_blobs(self, code_blob: str) -> str:
        """Parses the LLM's output to get any code blob inside. Will return the code directly if it's code."""
        pattern = r"```(?:sql)?\n(.*?)\n```"
        matches = re.findall(pattern, code_blob, re.DOTALL)
        if len(matches) == 0:
            try:  # Maybe the LLM outputted a code blob directly
                sqlglot.parse(code_blob)
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
                        ```sql
                        SELECT final_answer(/* Your SQL code here */);
                        ```<end_code>""".strip())
            raise ValueError(f"""Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
                    Here is your code snippet:
                    {code_blob}
                    Make sure to include code with the correct pattern, for instance:
                    Thoughts: Your thoughts
                    Code:
                    ```sql
                    # Your SQL code here
                    ```<end_code>""".strip())
        return fix_final_answer_code("\n\n".join(match.strip() for match in matches))

__all__ = ["evaluate_sql_code", "LocalPostgresInterpreter"]
