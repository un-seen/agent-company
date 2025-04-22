import re
import copy
import sqlglot
from datetime import datetime, date
from psycopg2 import sql
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import traceback
import numpy as np
import pandas as pd
from agentcompany.mcp.base import ModelContextProtocolImpl
import logging
from agentcompany.driver.dict import dict_rows_to_markdown_table
import psycopg2
from urllib.parse import urlparse
from psycopg2.extras import RealDictCursor
from agentcompany.extensions.environments.exceptions import InterpreterError, ERRORS
from agentcompany.extensions.environments.base import ExecutionEnvironment, Observations

logger = logging.getLogger(__name__)


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


def parse_function_call(call_str, state: dict) -> Tuple[List[str], str]:
    """
    Parses a string of the form FUNCTION_NAME(args) and returns [function_name, arg_values].

    Parameters:
        call_str (str): The string representing the function call.
        state (dict): Dictionary containing values for argument substitution.

    Returns:
        list: A list containing [function_name, arg_values].
    """
    pattern = r'(\w+)\(([^)]+)\)'
    match = re.match(pattern, call_str)

    if match:
        func_name, args_str = match.groups()
        args = [arg.strip() for arg in args_str.split(',')]
        resolved_args = []
        for arg in args:
            # Handle cases with dot notation (e.g., object.attr)
            if '.' in arg:
                _, arg_name = arg.split('.', 1)
            else:
                arg_name = arg

            # Replace arg with state[arg_name] if present in state
            resolved_args.append(state.get(arg_name, arg_name))

        return func_name.lower(), resolved_args, args_str
    else:
        return None, None, None


def evaluate_ast(pg_conn, node, state, static_tools: Dict[str, ModelContextProtocolImpl]) -> Tuple[List[dict], List[Any]]:
    # Check if the expression is a sqlglot SELECT statement.
    # (sqlglot returns expressions from the sqlglot.exp module;
    # adjust the type check if needed.)
    function_call_list = []
    if isinstance(node, sqlglot.exp.Select) or isinstance(node, sqlglot.exp.Insert) or isinstance(node, sqlglot.exp.Update) or isinstance(node, sqlglot.exp.Delete) or isinstance(node, sqlglot.exp.Create) or isinstance(node, sqlglot.exp.TruncateTable) or isinstance(node, sqlglot.exp.Drop):
        # Convert the AST to a Postgres-compatible SQL string.
        # Check if NODE uses an MCP server. if yes then call the server and replace the node subtree with the result.
        for idx, statement in enumerate(node.expressions):
            function_name, function_arguments, args_str = parse_function_call(str(statement.this), state)
            if function_name is not None and function_name in static_tools:
                function_call_list.append((function_name, function_arguments))
                node.args["expressions"] = node.expressions[:idx] + [" " + args_str + " "] + node.expressions[idx+1:]
        sql_query = node.sql(dialect="postgres")
        try:
            result = []
            with pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                result = cursor.execute(sql_query)
                if isinstance(node, sqlglot.exp.Select):
                    result = cursor.fetchall()     
                else:
                    pg_conn.commit()               
            return result, function_call_list
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error executing SQL: {error_msg}")
            raise InterpreterError(f"Error executing SQL: {error_msg}")
    else:
        raise InterpreterError(f"Unsupported AST node type: {type(node)}")
        
        
def evaluate_sql_code(
    pg_conn,
    code: str,
    state: Dict[str, Any],
    static_tools: Optional[Dict[str, Callable]] = None,
    max_count: Optional[int] = None
) -> List[dict]:
    """
    Evaluate a sql expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (str): The code to evaluate.
        static_tools (Optional[Dict[str, Callable]]): The functions that may be called during the evaluation.
        state (Optional[Dict[str, Any]]): A dictionary mapping variable names to values.
        authorized_imports (List[str]): List of modules that are allowed to be imported.
        max_count (int): Maximum number of items to return from a SELECT statement.
    """
    if not (code.lower().startswith("drop") or code.lower().startswith("select") or code.lower().startswith("create") or code.lower().startswith("insert") or code.lower().startswith("update") or code.lower().startswith("delete")):
        code = f"SELECT {code}"
    
    if code.lower().startswith("select") and max_count != None:
        code += f" LIMIT {max_count};"
    print(f"Executing SQL code: {code}")
    try:
        expression = sqlglot.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Failed to parse code to create ast on line {e.lineno} due to: {type(e).__name__}\n"
            f"{e.text}"
            f"{' ' * (e.offset or 0)}^\n"
            f"Error: {str(e)}"
        )
    
    static_tools = static_tools.copy() if static_tools is not None else {}
    task = state.get("task", None)
    try:
        for node in expression:
            outputs, function_call_list = evaluate_ast(pg_conn, node, state, static_tools)
            if len(function_call_list) > 0 and outputs is not None:
                # function_call_output = []
                function_output = {}
                for function_call in function_call_list:
                    function_name, function_arguments = function_call
                    function_exec = static_tools[function_name]
                    function_output[function_name] = function_exec(task, function_arguments, outputs)
                outputs_copy = copy.deepcopy(outputs)
                for idx in range(len(outputs_copy)):
                    item = outputs_copy[idx]
                    for function_name in function_output:
                        try:
                            value = function_output[function_name][idx]    
                            if isinstance(value, np.ndarray):
                                value = value.tolist()
                                if len(value) == 1:
                                    value = value[0]
                            item[function_name] = value
                        except:
                            item[function_name] = None
                outputs = outputs_copy
        return outputs
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Code execution failed at node '{node}' in code '{code}' due to exception:\n{error_trace}"
        raise InterpreterError(error_msg)


def list_sql_tables(sql: str) -> list[str]:
    """
    Return every table referenced in an SQL string using sqlglot's parser.

    Parameters
    ----------
    sql : str
        One or more SQL statements (any dialect that sqlglot supports).

    Returns
    -------
    list[str]
        Table names (schema‑qualified when present) in the order they appear,
        duplicates removed case‑insensitively.
    """
    # sqlglot.parse returns a list in case the string contains ≥1 statement
    statements = sqlglot.parse(sql, error_level="ignore")

    seen, ordered = set(), []
    for stmt in statements:
        # Walk the AST: every Table node = physical table reference
        for tbl in stmt.find_all(sqlglot.exp.Table):
            # Build fully‑qualified name: [catalog].[db].[table] (skip Nones)
            full_name = ".".join(
                part for part in (tbl.catalog, tbl.db, tbl.name) if part
            )
            key = full_name.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(full_name)

    return ordered


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return obj


class PostgresSqlInterpreter(ExecutionEnvironment):
    
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
        self.state = {}
        self.additional_authorized_imports = additional_authorized_imports
        self.pg_config = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password
        }
        self.pg_conn = psycopg2.connect(**self.pg_config)
        self.sql_schema = []
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
        # Add base trusted tools to list
        self.static_tools = mcp_servers
        super().__init__(session_id=session_id, mcp_servers=mcp_servers)
        
    def reset_connection(self):
        self.pg_conn.close()
        self.pg_conn = psycopg2.connect(**self.pg_config)

    def get_view_list(self, code_action: str):
        code_action = self.parse_code_blob(code_action)
        logger.info(f"Input Code Action: {code_action}")
        table_list: List[str] = list_sql_tables(code_action)
        logger.info(f"Table list: {table_list}")
        view_list = []
        for table in table_list:
            if table.lower().endswith("_view"):
                view_list.append(table)
        logger.info(f"View list: {view_list}")
        return view_list
    
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
        max_count = None
        tupled_rows = evaluate_sql_code(
            self.pg_conn,
            code_action,
            state=self.state,
            static_tools=self.static_tools,
            max_count=max_count
        )
        logs = self.state.get("logs", "")
        # Return the result as a list of dictionaries
        return tupled_rows or [], logs, False
        
    def attach_variables(self, variables: dict):
        self.state.update(variables)

    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        self.static_tools.update(mcp_servers)

    def parse_function(self, code_blob: str) -> Dict[str, Callable]:
        raise NotImplementedError("parse_function not implemented")
                                  
    def parse_code_blob(self, code_blob: str) -> str:
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

    def parse_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Escape a Python value for safe inclusion as a SQL string literal in an INSERT statement.
        
        - None → NULL
        - single quotes ' → ''
        - backslashes \ → \\
        - null bytes removed
        - newlines \n → \\n, tabs \t → \\t, carriage returns removed
        
        Transforms value in context into a string ready to drop into SQL, including the surrounding single quotes
        (except for NULL, which is unquoted).
        
        Args:
            context (Dict[str, Any]): The context dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the transformed context.
        """
        # Extract relevant information from the context
        parsed_context = {}
        for key, value in context.items():
            if value is None:
                parsed_context[key] = "NULL"
            elif isinstance(value, str):
                # Convert to string
                s = str(value)
                # Remove null bytes
                s = s.replace('\x00', '')
                # Escape backslashes
                s = s.replace('\\', '\\\\')
                # Escape single quotes by doubling them
                s = s.replace("'", "''")
                # Normalize whitespace
                s = s.replace('\r', '')
                s = s.replace('\n', '\\n')
                s = s.replace('\t', '\\t')
                parsed_context[key] = s
            else:
                parsed_context[key] = value
        return parsed_context
    
    def get_storage_id(self, next_step_id: int) -> str:
        return f"{self.session_id}_temp_storage_{next_step_id}"
    
    def set_storage(self, next_step_id: int, code_action: str, observations: List[Dict[str, Any]] = None):
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
            
            if observations:
                # Collect all unique keys from all observations
                all_keys = set()
                for obs in observations:
                    all_keys.update(obs.keys())
                
                if not all_keys:
                    raise ValueError("Observations contain no keys to create columns")
                
                # Create column definitions safely
                columns = [sql.Identifier(key) for key in all_keys]
                column_defs = sql.SQL(', ').join(
                    sql.SQL("{} JSONB").format(col) for col in columns
                )
                data = json.dumps(observations, default=json_serial)
                print(f"Storage Data: {data}")
                # Create view with dynamic columns
                create_view = sql.SQL("""
                    CREATE VIEW {table} AS
                    SELECT *
                    FROM jsonb_to_recordset({data}::JSONB) AS t({columns})
                """).format(
                    table=sql.Identifier(temp_table_name),
                    columns=column_defs,
                    date=sql.SQL(data)
                )
                
                # Execute with observations JSON
                cur.execute(create_view, ())
            else:
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
    
__all__ = ["evaluate_sql_code", "PostgresSqlInterpreter"]
