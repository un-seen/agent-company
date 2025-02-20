import inspect
import time
from collections import deque
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
import json
from agentcompany.driver.surreal_executor import SurrealExecutor

from agentcompany.driver.monitoring import (
    AgentLogger,
    LogLevel,
)

from redis import Redis
import os

from agentcompany.driver.memory import (
    ActionStep,
    AgentMemory,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    ToolCall,
)
from agentcompany.driver.types import AgentImage, handle_agent_output_types
from agentcompany.driver.utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    parse_thought,
    parse_python_code_blobs,
    parse_json_tool_call,
    truncate_content,
)
from agentcompany.driver.default_tools import TOOL_MAPPING, FinalAnswerTool
from agentcompany.driver.local_bash_executor import LocalBashInterpreter
from agentcompany.driver.local_python_executor import (
    BASE_BUILTIN_MODULES,
    LocalPythonInterpreter,
    fix_final_answer_code,
)
from agentcompany.driver.models import (
    ChatMessage,
    MessageRole,
)
import re
from agentcompany.driver.agents import MultiStepAgent
from agentcompany.driver.tools import (
    DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
    Tool,
    get_tool_description_with_args,
)
from agentcompany.driver.types import AgentType


logger = getLogger(__name__)

YELLOW_HEX = "#d4b702"


def parse_graphql_code_blob(code_blob: str) -> str:
    """
    Parses the LLM's output to extract any GraphQL code blob inside.
    If the code blob is already a valid GraphQL query (e.g. starts with 'query', 'mutation', or '{'),
    it returns the code directly. Otherwise, it searches for a block of text wrapped in triple backticks
    (optionally with a 'graphql' tag) and returns the extracted code.

    Args:
        code_blob (str): The string output from the LLM containing the GraphQL code.

    Returns:
        str: The extracted GraphQL code.

    Raises:
        ValueError: If no valid GraphQL code block can be found.
    """
    # Regex to capture GraphQL code wrapped in triple backticks (with optional 'graphql' language tag)
    pattern = r"```(?:graphql)?\n(.*?)\n```"
    matches = re.findall(pattern, code_blob, re.DOTALL)

    if len(matches) == 0:
        # If no code block is found, check if the entire blob looks like GraphQL code
        trimmed = code_blob.strip()
        if (
            trimmed.startswith("query")
            or trimmed.startswith("mutation")
            or trimmed.startswith("{")
        ):
            return trimmed
        raise ValueError(
            f"GraphQL code blob is invalid because the regex pattern {pattern} was not found in the provided code_blob:\n{code_blob}"
        )

    # Return the extracted GraphQL code(s), joined with double newlines if multiple blocks were found
    return "\n\n".join(match.strip() for match in matches)


SURREAL_GRAPHQL_CODE_SYSTEM_PROMPT = """
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'GraphQL:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning toward solving the task.
Then, in the 'GraphQL:' sequence, you should write a valid GraphQL query. The GraphQL sequence must end with a ````<end_query>` marker.
During each intermediate step, you can use a query (for example, using a field like `log`) to output any important information you will then need.
These outputs will then appear in the 'Observation:' field, which will be available as input for the next step.

Here are a few notional query examples:
Task: "Fetch Current Weather"
Thought:
I need to retrieve the current weather information for New York. I will query the currentWeather field with the city name as an argument and request the temperature and condition.

```graphql
currentWeather(city: "New York") {
  temperature
  condition
}

```<end_query>

Observation:
The response shows a temperature of 72°F and a condition of "Clear".

Task: "Retrieve a List of Recent News Articles"
Thought:
I want to display recent news articles from a hypothetical news feed. I will query the newsFeed field, requesting each article’s title, author, and publication date.

```graphql
newsFeed(limit: 5) {
  title
  author
  publishedAt
}
```<end_query>

Observation:
The response contains five articles with their titles, authors, and dates such as "Breaking News: Market Rally", "Tech Innovations Today", etc.

---

Task: "Calculate the Average Test Score for Grade 10 Students"
Thought:
I need to determine the average test score for students in grade 10. Assuming our schema provides an averageScore field that accepts a grade parameter, I will query it directly.

```graphql
averageScore(grade: 10)
```<end_query>

Observation:
The response indicates that the average score for grade 10 is 85.

---

Task: "Fetch Detailed Profile Information for a Specific Student"
Thought:
I want to fetch a detailed profile of a student with a known ID. I will query the studentProfile field using the student ID and request the name, grade, and a list of test scores.

```graphql
studentProfile(id: "student123") {
  name {
    first
    last
  }
  grade
  testScores {
    subject
    score
  }
}
```<end_query>

Observation:
The response returns the student's first and last name, grade level, and an array of test scores for various subjects.

---

Task: "Plan a Trip Itinerary Based on Destination and Interests"
Thought:
I need to generate a trip itinerary for a user visiting Paris with interests in museums and food. I will query the tripPlanner field with these parameters and request the itinerary details.

```graphql
tripPlanner(destination: "Paris", interests: ["museums", "food"]) {
  itinerary {
    day
    activity
  }
}
```<end_query>

Observation:
The response returns an itinerary with activities such as "Visit the Louvre on Day 1" and "Food tour in Montmartre on Day 2".

---

Task: "Retrieve Stock Price Information"
Thought:
I want to obtain the latest stock price for a company with symbol "ACME". I will query the stockPrice field with the symbol and request the current price and change percentage.

```graphql
stockPrice(symbol: "ACME") {
  currentPrice
  changePercent
}
```<end_query>

Observation:
The response indicates that ACME’s current price is $123.45 with a change of +2.3%.

---

Above examples use notional fields that might not exist for you. Do not mention any field besides these fields in the database schema:

{{graphql_schema}}

{{managed_agents_descriptions}}

Here are the rules you should always follow to solve your task:

Always provide a 'Thought:' sequence, and a 'GraphQL:' sequence that ends with ```<end_query>.
Use only variables that you have defined.
Never create any notional variables in our queries.
Don't give up! You're in charge of solving the task, not providing directions to solve it.
Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""

def simple_data_type(label: str):
    if label == "str":
      return "string"
    elif label == "dict":
      return "object"
    else:
      return label


class SurrealDBAgent(MultiStepAgent):
    """
    In this agent, the LLM writes graphql queries to fetch data as per user prompt

    Args:
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        base_url (`str`): Base URL of the GraphQL server.
        namespace (`str`): Namespace of the GraphQL server.
        database (`str`): Database of the GraphQL server.
        system_prompt (`str`, *optional*): System prompt that will be used to generate the agent's actions.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        base_url: str,
        namespace: str,
        database: str,
        system_prompt: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = SURREAL_GRAPHQL_CODE_SYSTEM_PROMPT

        self.namespace = namespace
        self.database = database
        

        self.surreal_executor = SurrealExecutor(
            base_url=base_url,
            namespace=namespace,
            database=database,
        )

        data = self.surreal_executor("INFO FOR DB", {}, "sql")
        tables = data["tables"].keys()
        schema = {}
        define_field_statement = []
        for table in tables:
            table_sample = self.surreal_executor(f"SELECT * FROM {table} LIMIT 1", {}, "sql")
            if table_sample is not None and isinstance(table_sample, list) and len(table_sample) > 0:
                table_sample = table_sample[0]
                schema[table] = {}
                for key in table_sample.keys():
                    if key == "id":
                        continue
                    sample_value = self.surreal_executor(f"SELECT {key} FROM {table} WHERE {key}!=NONE AND {key}!=NULL LIMIT 1", {}, "sql")
                    print(f"QUERY: SELECT {key} FROM {table} WHERE {key}!=NONE AND {key}!=NULL LIMIT 1")
                    sample_value = sample_value[0][key]
                    if sample_value is None:
                        continue
                    sample_type = type(sample_value).__name__
                    simple_sample_type = simple_data_type(sample_type)
                    define_field_statement.append(f"DEFINE FIELD IF NOT EXISTS {key} ON TABLE {table} TYPE {simple_sample_type}")
                    schema[table][key] = f"sample: {sample_value}"
        self.surreal_executor("DEFINE CONFIG GRAPHQL TABLES AUTO", {}, "sql")
        for statement in define_field_statement:
            print(statement)
            self.surreal_executor(statement, {}, "sql")
        self.schema = schema
        super().__init__(
            name="surreal_graphql",
            tools=[],
            model=model,
            system_prompt=system_prompt,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        
    
    def initialize_system_prompt(self):
        self.system_prompt = super().initialize_system_prompt()
        
        def get_table_str(name: str):
            return f"""
            Table: {name}
            Fields:
            {"\n".join(self.schema[name].keys())}
            """
        database_str = f"""{"\n".join([get_table_str(name) for name in self.schema.keys()])}"""
        schema_str = f"""
        Accessible Data:
        {"\n".join(self.schema.keys())}
        {database_str}
        """
        self.system_prompt = self.system_prompt.replace("{{graphql_schema}}", schema_str)
        
        return self.system_prompt

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
            code_action = fix_final_answer_code(parse_graphql_code_blob(model_output))
        except Exception as e:
            error_msg = (
                f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            )
            raise AgentParsingError(error_msg, self.logger)

        log_entry.tool_calls = [
            ToolCall(
                name="surrealgraphql_interpreter",
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
            output = self.surreal_executor(
                code_action,
                self.state,
                "graphql"
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
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        log_entry.action_output = output
        return output if is_final_answer else None