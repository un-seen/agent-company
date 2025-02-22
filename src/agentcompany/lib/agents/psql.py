from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
import json
import re
import psycopg2
from psycopg2.extras import RealDictCursor

# Import agent-specific errors and utilities
from agentcompany.driver.monitoring import AgentLogger, LogLevel
from agentcompany.driver.local_python_executor import fix_final_answer_code
from agentcompany.driver.models import ChatMessage
from agentcompany.driver.agents import MultiStepAgent
from agentcompany.driver.utils import (
    AgentExecutionError,
    AgentGenerationError,
    AgentParsingError,
    truncate_content,
)
from agentcompany.driver.memory import (
    ActionStep,
    ToolCall
)

logger = getLogger(__name__)
YELLOW_HEX = "#d4b702"

def parse_psql_code_blob(code_blob: str) -> str:
    """
    Parses the LLM's output to extract any PSQL code blob.
    It searches for a block of text wrapped in triple backticks (optionally tagged with 'psql' or 'sql')
    and returns the extracted code. If no such block is found, it checks whether the entire blob looks like
    a valid SQL query (e.g. starts with SELECT, INSERT, etc.).
    
    Args:
        code_blob (str): The string output from the LLM containing the PSQL code.
    
    Returns:
        str: The extracted PSQL code.
    
    Raises:
        ValueError: If no valid PSQL code block can be found.
    """
    pattern = r"```(?:psql|sql)?\n(.*?)\n```"
    matches = re.findall(pattern, code_blob, re.DOTALL)
    if len(matches) == 0:
        trimmed = code_blob.strip()
        if (trimmed.lower().startswith("select") or 
            trimmed.lower().startswith("insert") or 
            trimmed.lower().startswith("update") or 
            trimmed.lower().startswith("delete")):
            return trimmed
        raise ValueError(
            f"PSQL code blob is invalid because the regex pattern {pattern} was not found in the provided code_blob:\n{code_blob}"
        )
    return "\n\n".join(match.strip() for match in matches)

def extract_table_id(s: str) -> str:
    """
    Extracts and returns the table id from a given string.
    Supports:
      - RecordID("person", 0) -> returns "person"
      - dicom_instance:1.2.826... -> returns "dicom_instance"
      - Plain strings (e.g., "dicom") return as is.
    """
    match = re.search(r'RecordID\("([^"]+)"', s)
    if match:
        return match.group(1)
    match = re.search(r'^([^:]+):', s)
    if match:
        return match.group(1)
    return s

# Updated system prompt with sample test cases.
PSQL_CODE_SYSTEM_PROMPT = """
You are an expert assistant who can solve any task using SQL code blobs for PostgreSQL. You will be given a task to solve as best you can.
To solve the task, you must plan forward in a series of steps, in a cycle of 'Thought:', 'PSQL:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning toward solving the task.
Then, in the 'PSQL:' sequence, you should write a valid PostgreSQL query. The PSQL sequence must end with a ```<end_query> marker.
During each intermediate step, you can use queries (for example, for logging or retrieving intermediate results) to output any important information you will then need.
These outputs will then appear in the 'Observation:' field, which will be available as input for the next step.

Here are a few notional query examples:

Task: "Retrieve All Users Over 30"
Thought:
I need to fetch all user records from the 'users' table where the age is greater than 30.
I will write a SELECT query that retrieves all fields from 'users' with a condition on the 'age' column.

```psql
SELECT * FROM users WHERE age > 30;
```<end_query>

Observation:
The query returns all records from the 'users' table where age is greater than 30.

---

Task: "Update Order Status"
Thought:
I need to update the 'orders' table to set the status to 'shipped' for orders that have a non-null shipment date.
I will write an UPDATE query that sets the status accordingly.

```psql
UPDATE orders SET status = 'shipped' WHERE shipment_date IS NOT NULL;
```<end_query>

Observation:
The query updates the status for all orders with a shipment date.

---

Task: "Delete Old Logs"
Thought:
I need to delete records from the 'logs' table that are older than 90 days.
I will write a DELETE query using a date condition.

```psql
DELETE FROM logs WHERE log_date < NOW() - INTERVAL '90 days';
```<end_query>

Observation:
The query deletes all log records older than 90 days.

---

Below are the accessible data details:
{{psql_schema}}

Do not introduce any fields that do not exist in the database schema.
Use only variables that you have defined.
Always double-quote column names and table names in your queries.
Use OFFSET and LIMIT to only fetch 100 rows at a time.
Never create any notional variables in your queries.
Don't give up! You are in charge of solving the task, not providing directions to solve it.

{{managed_agents_descriptions}}

Here are the rules you should always follow to solve your task:

Always provide a 'Thought:' sequence, and a 'GraphQL:' sequence that ends with ```<end_query>.
Use only variables that you have defined.
Never create any notional variables in our queries.
Don't give up! You're in charge of solving the task, not providing directions to solve it.
Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""


class PsqlAgent(MultiStepAgent):
    """
    An agent that reads its schema from PostgreSQL and executes PSQL queries.
    The agent retrieves the schema by querying PostgreSQL’s information_schema and then 
    updates its system prompt to include the accessible data.
    """
    def __init__(
        self,
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        pg_config: Dict[str, Any],
        system_prompt: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = PSQL_CODE_SYSTEM_PROMPT

        # Establish PostgreSQL connection and load schema information.
        self.pg_conn = psycopg2.connect(**pg_config)
        self.schema = {}
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = [row["table_name"] for row in cur.fetchall()]
            for table in tables:
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;",
                    (table,)
                )
                table_schema = {row["column_name"]: row["data_type"] for row in cur.fetchall()}
                self.schema[table] = table_schema

        super().__init__(
            name="psql_agent",
            tools=[],
            model=model,
            system_prompt=system_prompt,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
    
    def initialize_system_prompt(self):
        """
        Inserts the inferred PostgreSQL schema into the system prompt.
        """
        self.system_prompt = super().initialize_system_prompt()
        def get_table_str(name: str):
            fields = "\n".join(self.schema[name].keys())
            return f"Table: {name}\nFields:\n{fields}\n"
        schema_str = "\n".join(get_table_str(name) for name in self.schema.keys())
        self.system_prompt = self.system_prompt.replace("{{psql_schema}}", schema_str)
        return self.system_prompt

    def pg_executor(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> Any:
        """
        Executes the given PSQL query on PostgreSQL and returns the results as a list of dicts.
        """
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, **params)
            return cur.fetchall()

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages.copy()

        # Add new step in logs
        log_entry.model_input_messages = memory_messages.copy()
        try:
            additional_args = (
                {"grammar": self.grammar} if self.grammar is not None else {}
            )
            chat_message: ChatMessage = self.model(
                self.input_messages,
                stop_sequences=["<end_query>", "Observation:"],
                **additional_args,
            )
            log_entry.model_output_message = chat_message
            model_output = chat_message.content
            log_entry.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(
                f"Error in generating model output:\n{e}", self.logger
            ) from e

        self.logger.log(
            Group(
                Rule(
                    f"[italic]Output message of the LLM ({self.name}):",
                    align="left",
                    style="orange",
                ),
                Syntax(
                    model_output,
                    lexer="markdown",
                    theme="github-dark",
                    word_wrap=True,
                ),
            ),
            level=LogLevel.DEBUG,
        )

        try:
            code_action = fix_final_answer_code(parse_psql_code_blob(model_output))
        except Exception as e:
            error_msg = (
                f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            )
            raise AgentParsingError(error_msg, self.logger)

        log_entry.tool_calls = [
            ToolCall(
                name="psql_executor",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        # Execute
        self.logger.log(
            Panel(
                Syntax(
                    code_action,
                    lexer="graphql",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]Executing this code:",
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=LogLevel.INFO,
        )
        observation = ""
        is_final_answer = True
        try:
            # Parse
            output = self.pg_executor(
                code_action,
                self.state,
            )
            execution_outputs_console = []
            observation += "Execution logs:\n" + json.dumps(output)
        except Exception as e:
            error_msg = str(e)
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        log_entry.observations = observation
        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        log_entry.action_output = output
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        
        output = {"final_answer": output} if is_final_answer else {"answer": output}
        return output 