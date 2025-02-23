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
from agentcompany.driver.local_tfserving_executor import TFServingExecutor
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

def parse_json_code_blob(code_blob: str) -> str:
    """
    Parses the LLM's output to extract any JSON code blob.
    It searches for a block of text wrapped in triple backticks (optionally tagged with 'json')
    and returns the extracted JSON as a string. If no such block is found, it checks whether the entire
    blob is valid JSON.
    
    Args:
        code_blob (str): The string output from the LLM containing the JSON code.
    
    Returns:
        str: The extracted JSON code as a string.
    
    Raises:
        ValueError: If no valid JSON code block can be found or parsed.
    """
    pattern = r"```(?:json)?\n(.*?)\n```"
    matches = re.findall(pattern, code_blob, re.DOTALL)
    if len(matches) == 0:
        trimmed = code_blob.strip()
        try:
            # Attempt to parse the entire blob as JSON
            json.loads(trimmed)
            return trimmed
        except json.JSONDecodeError:
            raise ValueError(
                f"JSON code blob is invalid because the regex pattern {pattern} was not found and the input is not valid JSON:\n{code_blob}"
            )
    # Combine matches with newlines, ensuring each is valid JSON
    valid_json_blocks = []
    for match in matches:
        trimmed_match = match.strip()
        try:
            json.loads(trimmed_match)
            valid_json_blocks.append(trimmed_match)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON block found in code_blob:\n{trimmed_match}")
    return "\n\n".join(valid_json_blocks)

# Updated system prompt with sample test cases.
TFSERVING_CODE_SYSTEM_PROMPT = """
You are an expert assistant who can solve any visual task using Tensorflow Serving You will be given a task to solve as best you can.
To solve the task, you must plan forward in a series of steps, in a cycle of 'Thought:', 'JSON:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning toward solving the task.
Then, in the 'JSON:' sequence, you should write the valid tensorflow model and the input file url. The Predictor sequence must end with a ```<end_query> marker.

---
Task: "Predict sentiment on file: review.txt"
Thought:
The request mentions 'sentiment,' so I’ll use the sentiment model. 

```json
{
  "model_name": "sentiment",
  "file_url": "/tmp/review.txt"
}
```<end_query>

Observation:
The sentiment model returned predictions: [0.85, 0.15] (positive sentiment).

---
Task: "Classify image on file: cat.jpg"
Thought:
The request mentions 'image,' so I’ll use the image model. 

```json
{
  "model_name": "image",
  "file_url": "/tmp/cat.jpg"
}
```<end_query>

Observation:
The image model returned predictions: [0.95, 0.03, 0.02] (class: cat).

---
Task: "Transcribe audio on file: speech.wav"
Thought:
The request mentions 'audio,' so I’ll use the audio model for transcription. 

```json
{
  "model_name": "audio",
  "file_url": "/tmp/speech.wav"
}
```<end_query>

Observation:
The audio model returned: "Hello, how are you?" (transcribed text).

---
Task: "Detect objects in file: street.png"
Thought:
The request mentions 'detect objects,' so I’ll use the object detection model. 

```json
{
  "model_name": "object",
  "file_url": "/tmp/street.png"
}
```<end_query>

Observation:
The object model returned: [{"class": "car", "confidence": 0.92, "bbox": [100, 150, 200, 250]}].

---
Task: "Summarize text on file: article.txt"
Thought:
The request mentions 'summarize,' so I’ll use the summary model.

```json
{
  "model_name": "summary",
  "file_url": "/tmp/article.txt"
}
```<end_query>

Observation:
The summary model returned: "The article discusses AI advancements." (summary text).

---

Below are the accessible TFServing endpoints:
{{tfserving_schema}}

Here are the rules you should always follow to solve your task:

Always provide a 'Thought:' sequence, and a 'GraphQL:' sequence that ends with ```<end_query>.
Don't give up! You're in charge of solving the task, not providing directions to solve it.

{{managed_agents_descriptions}}

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""
from typing import TypedDict

class TfServingAgent(MultiStepAgent):
    """
    An agent that reads its schema from PostgreSQL and executes PSQL queries.
    The agent retrieves the schema by querying PostgreSQL’s information_schema and then 
    updates its system prompt to include the accessible data.
    """
    def __init__(
        self,
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        tfserving_config: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = TFSERVING_CODE_SYSTEM_PROMPT

        # Establish PostgreSQL connection and load schema information.
        tfserving_endpoints = {
            model_dict["model_name"]: model_dict["predict_endpoint"] for model_dict in tfserving_config
        }
        self.tfserving_executor = TFServingExecutor(tfserving_endpoints=tfserving_endpoints)
        self.tfserving_schema = {}
        for model_dict in tfserving_config:
            self.tfserving_schema[model_dict["model_name"]] = model_dict["description"]
        self.tfserving_config = tfserving_config
        super().__init__(
            name="tfserving_agent",
            tools=[],
            model=model,
            system_prompt=system_prompt,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
    
    def initialize_system_prompt(self):
        """
        Inserts the tfserving model schema into the system prompt.
        """
        self.system_prompt = super().initialize_system_prompt()
        tfserving_schema_str = "\n\n".join([f"{k}: {self.tfserving_schema[k]}" for k in self.tfserving_schema.keys()])
        self.system_prompt = self.system_prompt.replace("{{tfserving_schema}}", tfserving_schema_str)
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
            code_action = fix_final_answer_code(parse_json_code_blob(model_output))
        except Exception as e:
            error_msg = (
                f"Error in json parsing:\n{e}\nMake sure to provide correct json blobs."
            )
            raise AgentParsingError(error_msg, self.logger)

        log_entry.tool_calls = [
            ToolCall(
                name="tfserving_executor",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        # Execute
        self.logger.log(
            Panel(
                Syntax(
                    code_action,
                    lexer="json",
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
            code_action = json.loads(code_action)
            output, logs, is_final_answer = self.tfserving_executor(
                code_action,
                self.state,
            )
            output = {"final_answer": output } if is_final_answer else {"answer": output}
            output["description"] = self.tfserving_schema[code_action["model_name"]]
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
        return output 