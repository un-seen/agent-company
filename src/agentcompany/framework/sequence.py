import pandas as pd
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from agentcompany.llms.monitoring import (
    AgentLogger,
)
import re
import json
from redis import Redis
import os
from typing import TypedDict
from agentcompany.extensions.environments.base import ExecutionEnvironment, Observations
from agentcompany.llms.memory import ActionStep, JudgeStep, ValidateStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, FunctionCall, PlanningStepStatus
from agentcompany.driver.errors import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
)
from agentcompany.framework.ambient import AmbientPattern
from agentcompany.framework.prompt_template import populate_template
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.llms.base import (
    ChatMessage,
    BaseLLM
)
from agentcompany.llms.utils import (
    MessageRole
)

logger = getLogger(__name__)


class ReturnTypeItem(TypedDict):
    type: str
    description: str
    
class CodeOutput(TypedDict):
    SystemPrompt: str
    Environment: str
    ReturnType: Dict[str, ReturnTypeItem]
    
class SeriesStep(TypedDict):
    Source: str
    SourceEnvironment: str
    SourceEnvironmentConfig: Dict[str, Any]
    CodeOutput: Dict[str, Any]

class PromptTemplates(TypedDict):
    Series: List[SeriesStep]
    
class SequencePattern(AmbientPattern):
    """
    Agent class that takes as input a series and executes code in an environment to process every item in the series.
    The agent will keep track of the state of the environment and if there are new additions to the series it will
    call itself.
    Args:
        name (`str`): Name of the agent.
        interface_id (`str`): Interface ID of the agent.
        description (`str`): Description of the agent.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates (`PromptTemplates`, *optional*): Prompt templates for the agent.
        mcp_servers (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list`, *optional*): List of functions to call after each step.
        final_answer_checks (`list`, *optional*): List of Callables to run before returning a final answer for checking validity.
        final_answer_call (`Callable[[str, Optional[list[str]]], str]`, *optional*): Function to call to provide the final answer.
        max_steps (`int`, default `6`): Maximum number of steps the agent can take to solve the task.
    """
    
    description = "This is an agent implementing the sequence design pattern."
    name = "SequencePattern"
    inputs = "The task for the agent to solve in plain text english."
    output_type = "pandas.DataFrame"
    interface_id: str = None
    executor_environment: Dict[int, ExecutionEnvironment] = {}

    def __init__(
        self,
        name: str,
        session_id: str,
        interface_id: str,
        description: str,
        model: BaseLLM,
        prompt_templates: PromptTemplates,
        mcp_servers: List[ModelContextProtocolImpl],
    ):
        # Identifiers
        self.name = name
        self.interface_id = interface_id
        self.session_id = session_id
        # Super Init
        super().__init__()
        # LLM
        self.model = model
        self.description = description
        # Environment State
        self.state = {}
        # Prompt Templates
        self.prompt_templates = prompt_templates
        # MCP Servers
        self.setup_mcp_servers(mcp_servers)
        # Setup Series
        self.prompt_templates = prompt_templates
        # Logging
        self.setup_logger_and_memory()
        # Context
        self.task = None

    
    def change_and_load_function(self, task: str, environment: ExecutionEnvironment, existing_function_code: Any, code_variables: Dict[str, Any]) -> None:
        """
        Search for a function in the system.
        Args:
            function_name (`str`): Name of the function to search for.
        Returns:
            Callable: The function if found, None otherwise.
        """
        variables = {
            "task": task,
            "existing_function_code": existing_function_code,
            "environment_interface": environment.language,
        }
        variables.update(code_variables)
        function_update_prompt = self.prompt_templates["FunctionUpdate"]
        function_change_message = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        function_update_prompt,
                        variables=variables
                    )
                }
            ]
        }
        self.logger.log(
            text=function_change_message['content'][0]['text'],
            title=f"Change Function Prompt Input:"
        )
        try:
            code_output_message: ChatMessage = self.model([function_change_message])
            self.logger.log(
                text=code_output_message.content,
                title=f"Augmented_LLM_Output({self.interface_id}/{self.name}):"
            )
        except Exception as e:
            raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
        # Load code in execution environment
        none_in_backtick = re.search(r"```[a-z]*\n*None\n*`|return `None`", code_output_message.content)
        self.logger.log(
            title="None in Backtick:",
            text=f"None in backtick: {bool(none_in_backtick)}"
        )
        if none_in_backtick:
            try:
                callable_dict = environment.parse_function(existing_function_code)
                self.state.update(callable_dict)
                self.logger.log(title="Code Action:", text=", ".join(callable_dict.keys()))
            except Exception as e:
                raise AgentExecutionError(
                    f"Error in loading function:\n{e}\nMake sure to provide correct code blobs.",
                    self.logger,
                ) from e
        else:
            try:
                callable_dict = environment.parse_function(code_output_message.content)
                self.state.update(callable_dict)
                self.logger.log(title="Code Action:", text=", ".join(callable_dict.keys()))
                # Save function in Redis
                function_name = code_variables["Function"]
                function_key = self.get_function_id(function_name)
                self.redis_client.set(function_key, code_output_message.content)
                return
            except Exception as e:
                raise AgentExecutionError(
                    f"Error in loading function:\n{e}\nMake sure to provide correct code blobs.",
                    self.logger,
                ) from e
        
    def get_function_id(self, function_name: str) -> str:
        """
        Get the function key for a given function name.
        Args:
            function_name (`str`): Name of the function.
        Returns:
            str: The function key.
        """
        return f"{self.interface_id}/{self.name}/function/{function_name}"
    
    
    def generate_and_load_function(self, task: str, environment: ExecutionEnvironment, code_prompt: str, code_variables: Dict[str, Any]):
        # Create message
        variables = {
            "mcp_servers": self.mcp_servers,
            "task": task,
        }
        variables.update(code_variables)
        function_create_message = {
            "role": MessageRole.USER, 
            "content": [
                {
                    "type": "text", 
                    "text": populate_template(
                        code_prompt,
                        variables=variables,
                    )
                }
            ]
        }
        self.logger.log(
            text=function_create_message['content'][0]['text'], 
            title=f"Create Function Prompt Input:"
        )  
        function_name = variables["Function"]
        function_key = self.get_function_id(function_name)
        # Create function code
        try:
            code_output_message: ChatMessage = self.model([function_create_message])
            self.logger.log(
                text=code_output_message.content,
                title=f"Augmented_LLM_Output({self.interface_id}/{self.name}):"
            )
        except Exception as e:
            raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
        # Load code in execution environment
        try:
            callable_dict = environment.parse_function(code_output_message.content)
            self.state.update(callable_dict)
            self.logger.log(title="Code Action:", text=", ".join(callable_dict.keys()))
            # Save Function @ Key
            self.redis_client.set(function_key, code_output_message.content)
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            self.logger.log(title="Error:", text=error_msg)
            attempts += 1
            raise AgentExecutionError(
                "Error in generating function after 7 attempts.", self.logger
            ) from e
    
    def run(
        self,
        task: str,
        reset: bool = True,
        environment_variables: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            environment_variables (`dict`): Any environment variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!

        Example:
        ```py
        from agentcompany.runtime.multistep import MultiStepAgent
        agent = MultiStepAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        self.task = task
        if environment_variables is not None:
            self.state.update(environment_variables)
            for _, environment in self.executor_environment.items():
                environment.attach_variables(environment_variables)
        # Local Memory
        series_memory: List[Any] = []
        series_count = len(self.prompt_templates["Series"])
        current_series_index = 0
        while current_series_index < series_count:
            series = self.prompt_templates["Series"][current_series_index]
            if current_series_index == 0:
                # Setup Source and Fetch Series
                source  = series["Source"]
                source_environment_interface = series["SourceEnvironment"]
                source_environment_config = series["SourceEnvironmentConfig"] or {}
                source_environment = self.setup_environment(source_environment_interface, source_environment_config)
                source_output, _, _ = source_environment(source, self.state, return_type="dict")
                source_output: Dict[str, Any] = source_output
                series_memory.append(source_output)
            # Setup Code Environment
            code_environment_interface = series["CodeOutput"]["Environment"]
            code_environment_config = series["CodeOutput"]["EnvironmentConfig"] or {}
            code_environment = self.setup_environment(code_environment_interface, code_environment_config)
            # Check Compute Type
            prompt_based_compute = "Prompt" in series["CodeOutput"] and series["CodeOutput"]["Prompt"]
            constant_code_based_compute = "Constant" in series["CodeOutput"] and series["CodeOutput"]["Constant"]
            # If Prompt Provided Call LLM
            if prompt_based_compute:
                function_name = series["CodeOutput"]["Variables"]["Function"]
                function_key = self.get_function_id(function_name)
                code_variables = series["CodeOutput"]["Variables"]
                code_prompt = series["CodeOutput"]["Prompt"]
                # The user is asking to call a minted function
                # 1. Is the user referring to an existing function?
                existing_function_code = self.redis_client.get(function_key)
                if existing_function_code is not None:
                    existing_function_code = existing_function_code.decode("utf-8")
                    self.logger.log(title="Existing Function Code:", text=existing_function_code)
                    # 1.1 Is the user asking to change the function ?
                    # Call LLM to change and load function
                    self.change_and_load_function(task, code_environment, existing_function_code, code_variables)
                else:
                    self.generate_and_load_function(task, code_environment, code_prompt, code_variables)
            # Calculate Series Output
            series_output = []
            series_input = series_memory[current_series_index]
            for series_input_item in series_input:
                series_input_item: Dict[str, Any] = series_input_item
                if constant_code_based_compute:
                    code_action = populate_template(
                        series["CodeOutput"]["Constant"],
                        variables={k: json.dumps(v) if isinstance(v, dict) else v for k, v in series_input_item.items()},
                    )
                    self.logger.log(title="Code Action:", text=code_action)
                    item_output, _, _ = code_environment(code_action, additional_variables=self.state, return_type="dict")
                elif prompt_based_compute:
                    function_name = series["CodeOutput"]["Variables"]["Function"]
                    function_input_variables = [series_input_item[e["name"]] for e in series["CodeOutput"]["Variables"]["InputType"]]
                    function_input_variables = [f"'{e}'" if isinstance(e, str) else e for e in function_input_variables]
                    function_call_str = f"{function_name}({','.join(function_input_variables)})"
                    code_action = function_call_str
                    item_output, _, _ = code_environment(code_action=code_action, additional_variables=self.state, return_type="dict")
                    item_output: Dict[str, Any] = item_output
                    item_output.update(series_input_item)
                else:
                    raise ValueError("Either Constant or Prompt based Series Execution is required.")
                series_output.append(item_output)
            self.logger.log(title=f"Series [{current_series_index}] Output:", text=series_output)
            series_memory.append(series_output)
            current_series_index += 1
            
        return pd.DataFrame(series_memory[-1])                
    
    
    def forward(self, task: str) -> Any:
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)