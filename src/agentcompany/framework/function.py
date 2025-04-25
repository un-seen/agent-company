# TODO implement the function design pattern such that the input is a dict
# and the output is a dict
# The function takes as input a task, and input dict 
# then it returns a dict
# it maps a list to a list
# if item is not a list then converts it into a list and returns first entry of output list
# this framework is to be implemented for keras and vllm
# one keras is for likelihood scores etc
# vllm is to map png to a css code etc
# LLMs can self-evaluate outputs and rank multiple completions,
# enabling autonomous work on a single prompt for hours. 
# I thought this would be a major trend, but people still use the first randomly sampled 
# text the model happened to spit out

import pandas as pd
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from agentcompany.llms.monitoring import (
    AgentLogger,
)
import copy
from agentcompany.llms.base import (
    AugmentedLLM,
    
)
from typing_extensions import Literal
from agentcompany.llms.base import (
    ReturnType
)
from agentcompany.extensions.environments.postgres_sql_executor import PostgresSqlInterpreter
from redis import Redis
import os
from agentcompany.extensions.environments.base import ExecutionEnvironment, Observations
from agentcompany.llms.memory import VisionStep, JudgeStep, ValidateStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, PlanningStepStatus
from agentcompany.driver.errors import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
)
from agentcompany.driver.markdown import list_of_dict_to_markdown_table
from agentcompany.mcp.base import ModelContextProtocolImpl
from typing import TypedDict
from agentcompany.framework.prompt_template import ExecutionEnvironmentConfig, populate_template
from agentcompany.llms.base import (
    ChatMessage,
    Argument,
    BaseLLM
)
from agentcompany.llms.utils import (
    MessageRole
)

logger = getLogger(__name__)

type ActionType = Literal["final_answer", "skip", "execute"]

class Node(TypedDict):
    """
    Node for the Agent Flow
    """
    step: str
    agent: Union[str, None]
    action: Union[ActionType, None]
    out: Union[Literal["one_to_many", "one_to_one", "many_to_one"], None]
    out_id: Union[str, None]
    
class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """
    executor_environment: ExecutionEnvironmentConfig
    plan: List[Node]
    
    
class CodeChoiceDefinition(TypedDict):
    """
    Preprocess definition for the agent.

    Args:
        choice_id (`str`): Choice ID.
        python (`str`): Python code.
        sql (`str`): SQL code.
        tfserving (`str`): TensorFlow Serving code.
    """
    choice_id: str
    description: str
    keyword: List[str]
    python: Optional[str]
    sql: Optional[str]
    

class FunctionPattern(ModelContextProtocolImpl):
    """
    Agent class that takes as input a prompt and calls if required one or more agents to execute a vision.
    
    Args:
        name (`str`): Name of the agent.
        session_id (`str`): Session ID of the agent.
        interface_id (`str`): Interface ID of the agent.
        description (`str`): Description of the agent.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates (`PromptTemplates`, *optional*): Prompt templates for the agent.
    """
    
    description = "This is an agent implementing the function design pattern."
    name = "FunctionPattern"
    inputs = "The task for the agent to solve in plain text english."
    output_type = "pandas.DataFrame"
    interface_id: str = None
    
    def __init__(
        self,
        name: str,
        session_id: str,
        interface_id: str,
        description: str,
        model: BaseLLM,
        prompt_templates: PromptTemplates,
    ):
        # Identifiers
        self.name = name
        self.interface_id = interface_id
        self.session_id = session_id
        # LLM
        self.model = model
        # Prompt Templates
        self.prompt_templates = prompt_templates
        # State
        self.state = {}
        # Environment
        self.executor_environment_config = self.prompt_templates["executor_environment"]
        # Postgres Agent
        # self.postgres_agent = postgres_agent
        self.setup_environment()
        self.description = description
        # Logging
        verbosity_level: int = 1
        self.logger = AgentLogger(name, interface_id, level=verbosity_level, use_redis=True)
        # Storage Client
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
        # Context
        self.input_messages = None
        self.task = None
        # Vision
        self.plan_step = None
        # Memory
        self.memory = AgentMemory(name, interface_id)
        super().__init__()

    @property
    def logs(self):
        return [self.memory.system_prompt] + self.memory.steps

    def set_verbosity_level(self, level: int):
        self.logger.set_level(level)
    
    def write_memory_to_messages(
        self,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = []
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages())
        return messages
    
    def extract_argument(self, code_content: str, argument_list: list[Argument]) -> dict[str, str]:
        """
        Extracts the argument from the code content.
        """
        llm: AugmentedLLM = self.model
        output: Dict[str, Any] = llm.function_call(code_content, argument=argument_list)
        return output
    
    def execute_main_choice(self, model_input_messages: list[dict[str, any]], context: list[dict[str, any]]) -> Tuple[str, Any]:
        model_input_messages_str = "\n".join([msg["content"][0]["text"] for msg in model_input_messages])
        self.logger.log(text=model_input_messages_str, title=f"Augmented_LLM_Input({self.interface_id}/{self.name}):")
        try:
            code_output_message: ChatMessage = self.model(model_input_messages)
        except Exception as e:
            raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
        
        choice_id = code_output_message.content
        # TODO parse the choice id
        self.logger.log(text=choice_id, title=f"Main Choice ID ({self.interface_id}/{self.name}):")
        main_choice_list = self.prompt_templates["main_choice"]
        main_choice: CodeChoiceDefinition = None
        for item in main_choice_list:
            if item["choice_id"] in choice_id:
                main_choice = item
                break
        if main_choice is None:
            raise ValueError(f"Choice ID '{choice_id}' not found in main choices.")
        code_content = main_choice[self.executor_environment.language]
        self.logger.log(text=code_content, title=f"Main Function Choice ({self.interface_id}/{self.name}):")
        variables = {}
        print(f"main_choice: {main_choice}")
        if len(main_choice["argument"]) > 0:
            argument_list = main_choice["argument"]
            argument_dict = self.extract_argument(code_content, argument_list)
            print(f"argument_dict: {argument_dict}")
            variables.update(argument_dict)
        # TODO handle case when context has to render for multiple items individually
        # in a single run call
        if len(context) == 1:
            variables.update(self.executor_environment.parse_context(context[0]))
        # Populate the code content with the context
        code_action = populate_template(
            code_action,
            variables=variables
        )   
        self.logger.log(text=code_action, title=f"Code Output ({self.interface_id}/{self.name}):")
        try:
            code_action = self.executor_environment.parse_code_blob(code_content)
        except Exception as e:
            
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            error_msg = self.executor_environment.parse_error_logs(error_msg)
            raise AgentError(f"Error in code parsing:\n{e}", self.logger) from e
        
        try:
            observations, _, _ = self.executor_environment(
                code_action=code_action,
                additional_variables={
                    "context": context
                },
                return_type="pandas.DataFrame"
            )
        except Exception as e:
            error_msg = "Error in Code Execution: \n"
            if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
            error_msg += str(e)
            error_msg = self.executor_environment.parse_error_logs(error_msg)
            raise AgentError(f"Error in code execution:\n{e}", self.logger) from e

        return code_action, observations
    
    
    def run(
        self,
        task: str,
        inputs: Union[str, List[str]] = None,
        context: Union[str, List[str], List[Dict[str, Any]], Dict[str, Any]] = None, 
        reset: bool = True,
        environment_variables: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            environment_variables (`dict`): Any environment variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task.

        Example:
        ```py
        from agentcompany.runtime.multistep import MultiStepAgent
        agent = MultiStepAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        # Add Environment Variables
        if environment_variables is not None:
            self.state.update(environment_variables)
            
        if reset:
            self.memory.reset()
        self.memory.append_step(TaskStep(task=self.task))
        task: str = task
        inputs: List[str] = inputs
        if isinstance(context, dict):
            context = [context]
        context: List[Dict[str, Any]] = context
        
        if not isinstance(task, str) or not isinstance(inputs, (list, str)) or not isinstance(context, (list, dict)):
            raise ValueError("Task should be a string, inputs should be a list of strings, and context should be a list of dictionaries.")
        
        # Main
        # Input Message
        main_prompt = self.prompt_templates["main"]
        main_choice = self.prompt_templates["main_choice"]
        context_as_str = list_of_dict_to_markdown_table(context[:5]) if len(context) > 0 else ""
        input_message = populate_template(
            main_prompt,
            variables={
                "task": task,
                "inputs": inputs,
                "context_as_str": context_as_str,
                "main_choice": main_choice,
            }
        )
        # Model Input Messages
        model_input_messages = [
            {"role": "system", "content": [{"type": "text", "text": input_message}]},
        ]
        main_code, main_outputs = self.execute_main_choice(model_input_messages, context)
        return main_outputs
                
    def setup_environment(self):
        # Get class name from config
        interface_name = self.executor_environment_config["interface"]

        # Find all registered ExecutionEnvironment subclasses
        from agentcompany.extensions.environments.local_python_executor import LocalPythonInterpreter
        from agentcompany.extensions.environments.postgres_sql_executor import PostgresSqlInterpreter
        from agentcompany.extensions.environments.jupyter_python_executor import JupyterPythonInterpreter
        
        environment_classes = {cls.__name__: cls for cls in ExecutionEnvironment.__subclasses__()}        
        try:
            environment_cls = environment_classes[interface_name]
        except KeyError:
            available = list(environment_classes.keys())
            raise ValueError(
                f"Unknown execution environment '{interface_name}'. "
                f"Available implementations: {available}"
            ) from None

        # Instantiate the chosen class
        self.executor_environment = environment_cls(
            self.session_id,
            mcp_servers={},
            **self.executor_environment_config["config"]
        )
            
    
    def forward(self, 
                task: str, 
                inputs: Union[str, List[str]] = None, 
                context: Union[str, List[str], List[Dict[str, Any]], Dict[str, Any]] = None) -> Any:
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task, inputs=inputs, context=context)