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
To do so, you have been given access to a list of tools: these tools are basically GraphQL queries which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'GraphQL:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning toward solving the task and the tools that you want to use.
Then, in the 'GraphQL:' sequence, you should write the query in simple GraphQL. The GraphQL sequence must end with a `<end_query>` marker.
During each intermediate step, you can use a query (e.g. using a field like `log`) to output any important information you will then need.
These outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically GraphQL queries which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'GraphQL:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning toward solving the task and the tools that you want to use.
Then, in the 'GraphQL:' sequence, you should write the query in simple GraphQL. The GraphQL sequence must end with a `<end_query>` marker.
During each intermediate step, you can use a query (e.g. using a field like `log`) to output any important information you will then need.
These outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Thought: I will proceed step by step and use the following tools: use `document_qa` to find the oldest person in the document, then use `image_generator` to generate an image based on the answer.
GraphQL:
```graphql
query {
  answer: document_qa(document: $document, question: "Who is the oldest person mentioned?") {
    result
  }
  log(message: "Retrieved answer from document_qa")
}
```<end_query>

Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Thought: I will now generate an image showcasing the oldest person. GraphQL:

```graphql
query {
  image: image_generator(prompt: "A portrait of John Doe, a 55-year-old man living in Canada.") {
    url
  }
  final_answer(answer: $image)
}
```<end_query>

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Thought: I will use a GraphQL query to compute the result of the operation and then return the final answer using the final_answer tool. GraphQL:

```graphql
query {
  result: compute(operation: "5 + 3 + 1294.678") {
    value
  }
  final_answer(answer: $result.value)
}
```<end_query>

---
Task: "Answer the question in the variable question about the image stored in the variable image. The question is in French. You have been provided with these additional arguments, available as variables in your GraphQL query: { question: 'Quel est l'animal sur l'image?', image: 'path/to/image.jpg' }"

Thought: I will use the following tools: first, translator to translate the question into English, then image_qa to answer the question about the input image. GraphQL:

```graphql
query {
  translated_question: translator(question: $question, src_lang: "French", tgt_lang: "English") {
    result
  }
  log(message: "The translated question is: " + $translated_question.result)
  answer: image_qa(image: $image, question: $translated_question.result) {
    result
  }
  final_answer(answer: "The answer is " + $answer.result)
}
```<end_query>

Task: In a 1979 interview, Stanislaus Ulam discusses with Martin Sherwin about other great physicists of his time, including Oppenheimer. What does he say was the consequence of Einstein learning too much math on his creativity, in one word?

Thought: I need to find and read the 1979 interview of Stanislaus Ulam with Martin Sherwin. GraphQL:

```graphql
query {
  pages: data_search(query: "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein") {
    results
  }
  log(message: "Searched for interview pages")
}
```<end_query>
Observation: No result found for query "1979 interview Stanislaus Ulam Martin Sherwin physicists Einstein".

Thought: The query was maybe too restrictive and did not find any results. Let's try again with a broader query. GraphQL:

```graphql
query {
  pages: data_search(query: "1979 interview Stanislaus Ulam") {
    results
  }
  log(message: "Broader search yielded pages")
}
```<end_query>
Observation: Found 6 pages: Stanislaus Ulam 1979 interview

Ulam discusses Manhattan Project

(truncated)

Thought: I will read the first 2 pages to know more. GraphQL:

```graphql
query {
  page1: visit_webpage(url: "https://ahf.nuclearmuseum.org/voices/oral-histories/stanislaus-ulams-interview-1979/") {
    content
  }
  log(message: "Visited page 1")
  page2: visit_webpage(url: "https://ahf.nuclearmuseum.org/manhattan-project/ulam-manhattan-project/") {
    content
  }
  log(message: "Visited page 2")
}
```<end_query>
Observation: Manhattan Project Locations: Los Alamos, NM Stanislaus Ulam was a Polish-American mathematician. He worked on the Manhattan Project at Los Alamos and later helped design the hydrogen bomb. In this interview, he discusses his work at (truncated)

Thought: I now have the final answer: from the webpages visited, Stanislaus Ulam says of Einstein: "He learned too much mathematics and sort of diminished, it seems to me personally, it seems to me his purely physics creativity." Let's answer in one word. GraphQL:

```graphql
query {
  final_answer(answer: "diminished")
}
```<end_query>

---
Task: "Which city has the highest population: Guangzhou or Shanghai?"

Thought: I need to get the populations for both cities and compare them. I will use the tool data_search to get the population of both cities. GraphQL:

```graphql
query {
  guangzhou: data_search(query: "Guangzhou population") {
    results
  }
  shanghai: data_search(query: "Shanghai population") {
    results
  }
  log(message: "Retrieved populations for Guangzhou and Shanghai")
}
```<end_query>
Observation: Population Guangzhou: "Guangzhou has a population of 15 million inhabitants as of 2021." Population Shanghai: "26 million (2019)"

Thought: Now I know that Shanghai has the highest population. GraphQL:

```graphql
query {
  final_answer(answer: "Shanghai")
}
```<end_query>

---
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the tool data_search to get the age of the pope, and confirm that with a web_search tool. GraphQL:

```graphql
query {
  pope_age_wiki: data_search(query: "current pope age") {
    results
  }
  log(message: "Retrieved pope age from wiki")
  pope_age_search: web_search(query: "current pope age") {
    results
  }
  log(message: "Retrieved pope age from web search")
}
```<end_query>
Observation: Pope age: "The pope Francis is currently 88 years old."

Thought: I know that the pope is 88 years old. Let's compute the result using a GraphQL computation (assuming a tool compute exists). GraphQL:

```graphql
query {
  pope_computation: compute(expression: "88 ** 0.36") {
    value
  }
  final_answer(answer: $pope_computation.value)
}
```<end_query>
Above examples were using notional tools that might not exist for you. On top of performing computations in the GraphQL queries that you create, you only have access to these tools:

{{tool_descriptions}}

{{managed_agents_descriptions}}

Here are the rules you should always follow to solve your task:

Always provide a 'Thought:' sequence, and a 'GraphQL:' sequence that ends with <end_query>, else you will fail.
Use only variables that you have defined!
Always use the right arguments for the tools. DO NOT pass the arguments as a dictionary (for example, answer: wiki({query: "What is the place where James Bond lives?"})), but pass the arguments directly as in answer: wiki(query: "What is the place where James Bond lives?").
Take care to not chain too many sequential tool calls in the same query block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather, output results with a logging field to use them in the next block.
Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
Don't name any new variable with the same name as a tool: for instance, don't name a variable 'final_answer'.
Never create any notional variables in our queries, as having these in your logs will derail you from the true variables.
The state persists between query executions: so if in one step you've created variables or imported fragments, these will all persist.
Don't give up! You're in charge of solving the task, not providing directions to solve it.
Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""


class TFServingAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in a json format to a remote Rest API based serving engine.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        system_prompt (`str`, *optional*): System prompt that will be used to generate the agent's actions.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        # TODO fix this to receive dict tools
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        system_prompt: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        max_print_outputs_length: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = BASH_CODE_SYSTEM_PROMPT

        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        if "{{authorized_imports}}" not in system_prompt:
            raise ValueError("Tag '{{authorized_imports}}' should be provided in the prompt.")
        super().__init__(
            tools=tools,
            model=model,
            system_prompt=system_prompt,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                0,
            )

        all_tools = {**self.tools, **self.managed_agents}
        self.bash_executor = LocalBashInterpreter(
            all_tools,
            self.additional_authorized_imports,
            max_print_outputs_length=max_print_outputs_length,
        )

    def initialize_system_prompt(self):
        self.system_prompt = super().initialize_system_prompt()
        # TODO fix this security issue
        self.system_prompt = self.system_prompt.replace(
            "{{authorized_imports}}",
            "You can import from any package you want."
            if "*" in self.authorized_imports
            else str(self.authorized_imports),
        )
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
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            chat_message: ChatMessage = self.model(
                self.input_messages,
                stop_sequences=["<end_code>", "Observation:"],
                **additional_args,
            )
            log_entry.model_output_message = chat_message
            model_output = chat_message.content
            log_entry.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

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

        # Parse
        try:
            code_action = fix_final_answer_code(parse_python_code_blobs(model_output))
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        log_entry.tool_calls = [
            ToolCall(
                name="bash_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        # Execute
        self.logger.log(
            Panel(
                Syntax(
                    code_action,
                    lexer="python",
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
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.bash_executor(
                code_action,
                self.state,
            )
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation += "Execution logs:\n" + execution_logs
        except Exception as e:
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
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

