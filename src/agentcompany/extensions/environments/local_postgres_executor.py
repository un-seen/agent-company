
import re
import sqlglot
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import traceback
import numpy as np
import pandas as pd
import sqlglot.expressions
from agentcompany.mcp.base import ModelContextProtocolImpl
import logging
from agentcompany.driver.dict import dict_rows_to_markdown_table
import psycopg2
from urllib.parse import urlparse
from psycopg2.extras import RealDictCursor
from agentcompany.mcp.utils import truncate_content
from agentcompany.extensions.environments.exceptions import InterpreterError, ERRORS
from agentcompany.extensions.environments.base import ExecutionEnvironment, Observations

logger = logging.getLogger(__name__)

BASE_BUILTIN_MODULES = [
    "time",
    "unicodedata",
]

PRINT_OUTPUTS = ""
DEFAULT_MAX_LEN_OUTPUT = 50000


def connect_with_uri(uri):
    """Connects to a PostgreSQL database using a connection URI.

    Args:
        uri (str): The PostgreSQL connection URI.

    Returns:
        psycopg2.extensions.connection: A psycopg2 connection object if successful,
                                        None otherwise.
    """
    conn = None
    try:
        parsed_uri = urlparse(uri)
        dbname = parsed_uri.path[1:]  # Remove the leading '/'
        user = parsed_uri.username
        password = parsed_uri.password
        host = parsed_uri.hostname
        port = parsed_uri.port if parsed_uri.port else 5432  # Default PostgreSQL port

        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        return conn
    except psycopg2.Error as e:
        return None

def get_iterable(obj):
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, "__iter__"):
        return list(obj)
    else:
        raise InterpreterError("Object is not iterable")


def evaluate_ast(pg_conn, node, state, static_tools: Dict[str, ModelContextProtocolImpl], custom_tools, authorized_imports) -> List[dict]:
    # Check if the expression is a sqlglot SELECT statement.
    # (sqlglot returns expressions from the sqlglot.exp module;
    # adjust the type check if needed.)
    if isinstance(node, sqlglot.exp.Select) or isinstance(node, sqlglot.exp.Insert) or isinstance(node, sqlglot.exp.Update) or isinstance(node, sqlglot.exp.Delete):
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
            result = []
            with pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                result = cursor.execute(sql_query)
                if isinstance(node, sqlglot.exp.Select):
                    result = cursor.fetchall()     
                else:
                    pg_conn.commit()               
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
    max_count: Optional[int] = None
) -> List[dict]:
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
        max_count (int): Maximum number of items to return from a SELECT statement.
    """
    if not (code.lower().startswith("select") or code.lower().startswith("create") or code.lower().startswith("insert") or code.lower().startswith("update") or code.lower().startswith("delete")):
        code = f"SELECT {code}"
    
    if code.lower().startswith("select") and max_count != None:
        code += f" LIMIT {max_count};"
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
    
    try:
        for node in expression:
            result = evaluate_ast(pg_conn, node, state, static_tools, custom_tools, authorized_imports)            
        state["print_outputs"] = truncate_content(PRINT_OUTPUTS)
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Code execution failed at node '{node}' in code '{code}' due to exception:\n{error_trace}"
        raise InterpreterError(error_msg)
    
    
class LocalPostgresInterpreter(ExecutionEnvironment):
    
    language: str = "sql"
    
    def __init__(
        self,
        session_id: str,
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
        # IMPROVE: assert self.authorized imports are all installed 
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        # Add base trusted tools to list
        self.static_tools = mcp_servers
        super().__init__(session_id=session_id, mcp_servers=mcp_servers)
        
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
        
    def __call__(self, code_action: str, additional_variables: Dict, return_type: str = "string") -> Tuple[Union[List[dict], str], str, bool]:
        self.state.update(additional_variables)
        self.reset_connection()
        code_action = code_action.strip(";")
        logger.info(f"{self.__class__.__name__}.__call__ Return type: {return_type}")
        max_count = 10 if return_type == "string" else None
        tupled_rows = evaluate_sql_code(
            self.pg_conn,
            code_action,
            static_tools=self.static_tools,
            custom_tools=self.custom_tools,
            state=self.state,
            authorized_imports=self.authorized_imports,
            max_count=max_count
        )
        logs = self.state["print_outputs"]
        if return_type == "string":
            markdown_table = dict_rows_to_markdown_table(tupled_rows)
            return markdown_table, logs, False
        else:
            # Return the result as a list of dictionaries
            return tupled_rows, logs, False
        
    def attach_variables(self, variables: dict):
        self.state.update(variables)

    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        self.static_tools.update(mcp_servers)

    def parse_function(self, code_blob: str) -> Dict[str, Callable]:
        raise NotImplementedError("parse_function not implemented")
                                  
    def parse_code_blobs(self, code_blob: str) -> str:
        """Parses the LLM's output to get any code blob inside. Will return the code directly if it's code."""
        pattern = r"```(?:sql)?\n(.*?)\n```"
        matches = re.findall(pattern, code_blob, re.DOTALL)
        if len(matches) == 0:
            try:  # Maybe the LLM outputted a code blob directly
                sqlglot.parse(code_blob)
                return code_blob
            except SyntaxError:
                raise ValueError(f"""Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
                        Here is your code snippet:
                        {code_blob}
                        Make sure to include code with the correct pattern, for instance:
                        Thoughts: Your thoughts
                        Code:
                        ```sql
                        # Your SQL code here
                        ```<end_code>""".strip())
        return "\n".join(match.strip() for match in matches)

    def get_storage_id(self, next_step_id: int) -> str:
        return f"{self.session_id}_temp_storage_{next_step_id}"
    
    def set_storage(self, next_step_id: int, code_action: str):
        """
        Executes the provided SQL code_action, stores the result in a temporary table,
        and returns a list of column dictionaries with name, type, and sample values.

        Args:
            next_step_id (int): The ID of the next step, used to generate a unique temp table name.
            next_step (str): Description of the next step (unused in this implementation).
            code_action (str): SQL query whose result is to be saved as a temporary table.
            observations (str): Observations related to this step (unused in this implementation).
            feedback (str): Feedback related to this step (unused in this implementation).

        Returns:
            list: A list of dictionaries, each containing 'name', 'type', and 'sample_values' for a column.
        """
        temp_table_name = self.get_storage_id(next_step_id)
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Ensure the temp table is dropped if it exists
            cur.execute(f"DROP VIEW IF EXISTS {temp_table_name} CASCADE;")
            # Create the temp table with the result of the code_action query
            code_action = code_action.strip(';')
            # Write to environment state
            cur.execute(f"CREATE VIEW {temp_table_name} AS ({code_action});")
            self.pg_conn.commit()
            
            # Retrieve column names and data types from the temp table
            cur.execute(f"""
                SELECT 
                    a.attname AS column_name,
                    pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type
                FROM 
                    pg_catalog.pg_attribute a
                WHERE 
                    a.attrelid = '{temp_table_name}'::regclass 
                    AND a.attnum > 0 
                    AND NOT a.attisdropped
                ORDER BY a.attnum;
            """)
            columns = cur.fetchall()
            
            # Fetch up to 3 sample rows from the temp table
            cur.execute(f"SELECT * FROM {temp_table_name} LIMIT 3;")
            sample_rows = cur.fetchall()
            
            # Prepare the list of column dictionaries
            column_dicts = []
            for col_info in columns:
                col_name = col_info['column_name']
                col_type = col_info['data_type']
                samples = [row[col_name] for row in sample_rows]
                column_dicts.append({
                    'name': col_name,
                    'type': col_type,
                    'sample_values': samples
                })
            # Write to storage
            self.storage[next_step_id] = column_dicts
        
                
    def reset_storage(self):
        self.storage = {}
    
    def get_final_storage(self) -> pd.DataFrame:
        max_step_id = max(self.storage.keys())
        temp_table_name = self.get_storage_id(max_step_id)
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch up to 3 sample rows from the temp table
            cur.execute(f"SELECT * FROM {temp_table_name};")
            sample_rows = cur.fetchall()
            for row in sample_rows:
                for key, value in row.items():
                    if isinstance(value, np.ndarray):
                        row[key] = value.tolist()
        return pd.DataFrame(sample_rows)
            
    def get_storage(self, next_step_id: int) -> str:
        select_prompt = []
        info_prompt = []
        temp_table_name = self.get_storage_id(next_step_id)
        if next_step_id in self.storage:
            columns = self.storage[next_step_id]
            for col in columns:
                col_name = col['name']
                col_type = col['type']
                sample_value = ", ".join([item for item in col['sample_values'] if isinstance(item, str)])
                select_prompt.append(col_name)
                info_prompt.append(f"{col_name} ({col_type}), Example: {sample_value}")
            
        return f"""
        Code to get result of step {next_step_id} in {temp_table_name}:
            SELECT {', '.join(select_prompt)} FROM {temp_table_name};
            
        Information on the {temp_table_name} table:
            {', '.join(info_prompt)}
        """
    
__all__ = ["evaluate_sql_code", "LocalPostgresInterpreter"]
