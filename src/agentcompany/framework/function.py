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
from typing_extensions import Literal
from agentcompany.llms.base import (
    ReturnType
)
from agentcompany.extensions.environments.local_postgres_executor import LocalPostgresInterpreter
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
    
    def _validate_observations(self, step: str, previous_observations: List[Observations]) -> PlanningStepStatus:
        # Validate if next step is complete
        variables = {
            "role": self.description,
            "task": step,
            "observations": previous_observations,
        }
        message_prompt_plan = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["validate_observations"],
                        variables=variables,
                    ),
                }
            ],
        }
        self.logger.log(text=message_prompt_plan["content"][0]["text"], title="Validate Observations Input:")
        validate_answer_message: ChatMessage = self.model([message_prompt_plan])
        self.validate_step = ValidateStep([message_prompt_plan], validate_answer_message)
        self.logger.log(text=validate_answer_message.content, title="Validate Observations Output:")
        return self.validate_step.to_decision()
    
    def preprocess(
            self, 
            task: str,
            inputs: List[str] = None,
            context: List[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
        # Input Message
        # Convert inputs to a markdown table
        context_as_str = list_of_dict_to_markdown_table(context)
        preprocess_prompt = self.prompt_templates["preprocess"]
        input_message = populate_template(
            preprocess_prompt,
            variables={
                "task": task,
                "inputs": inputs,
                "context_as_str": context_as_str,
            }
        )
        # Hints
        input_message_lower = input_message.lower() 
        hints = self.prompt_templates["preprocess_hint"]
        filtered_hints = [
            hint for hint in hints
            if any(keyword.lower() in input_message_lower for keyword in hint.get("keyword", []))
        ]
        filtered_hints_str = f"""
        ## Hints
        
        {list_of_dict_to_markdown_table(filtered_hints)}
        """.strip()
        
        # Model Input Messages
        model_input_messages = [
            {"role": "system", "content": [{"type": "text", "text": input_message}]},
            {"role": "user", "content": [{"type": "text", "text": filtered_hints_str}]}
        ]
        model_input_messages_str = "\n".join([msg["content"][0]["text"] for msg in model_input_messages])
        self.logger.log(text=model_input_messages_str, title=f"Augmented_LLM_Input({self.interface_id}/{self.name}):")

        previous_environment_errors: List[Dict[str, Any]] = []
        # Loop
        while True:
            model_input_messages_with_errors = copy.deepcopy(model_input_messages)

            if previous_environment_errors:
                error_str = "\n\n".join([
                    f"Error encountered:\n{err['error']}\nProblematic code:\n{err['code']}"
                    for err in previous_environment_errors
                ])
                model_input_messages_with_errors.append(
                    {"role": "system", "content": [{"type": "text", "text": error_str}]}
                )    
            try:
                code_output_message: ChatMessage = self.model(model_input_messages_with_errors)
            except Exception as e:
                raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
            observations = None
            try:
                code_action = self.executor_environment.parse_code_blobs(code_output_message.content)
            except Exception as e:
                error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                error_msg = self.executor_environment.parse_error_logs(error_msg)
                previous_environment_errors.append({"code": code_output_message.content, "error": error_msg})
                continue
            
            self.logger.log(text=code_action, title=f"Code Output ({self.interface_id}/{self.name}):")
            try:
                observations, _, _ = self.executor_environment(
                    code_action=code_action,
                    additional_variables={
                        "inputs": pd.DataFrame(context)
                    },
                    return_type="pandas.DataFrame"
                )
                self.logger.log(text=observations, title=f"Output from code execution: {len(observations)} characters")
            except Exception as e:
                error_msg = "Error in Code Execution: \n"
                if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                    error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
                error_msg += str(e)
                error_msg = self.executor_environment.parse_error_logs(error_msg)
                self.logger.log(text=f"Code: {code_action}\n\nError: {error_msg}", title="Error in Code Execution:")
                previous_environment_errors.append({"code": code_action, "error": error_msg})
                continue
            
            judge_input_message = {
                "role": MessageRole.USER, 
                "content": [{
                    "type": "text", 
                    "text": populate_template(
                        self.prompt_templates["judge"],
                        variables={
                            "task": model_input_messages_str,
                            "code": code_action,
                            "observations": observations,
                        }
                    )
                }]
            }
            self.logger.log(text=judge_input_message["content"][0]["text"], title=f"Judge Input ({self.interface_id}/{self.name}):")
            judge_output_message: ChatMessage = self.model([judge_input_message])

            self.judge_step = JudgeStep([judge_input_message], judge_output_message)
            self.memory.append_step(self.judge_step)

            decision = self.judge_step.to_decision()
            feedback = self.judge_step.get_feedback_content()

            self.logger.log(text=self.judge_step.model_output_message.content, title=f"Judge Output ({self.interface_id}/{self.name}):")
            self.logger.log(text=decision, title=f"Judge Decision ({self.interface_id}/{self.name}):")

            if decision == "approve":
                break
            else:
                previous_environment_errors = [{"code": code_action, "error": feedback}]

        return observations
    
    
    def main(
            self, 
            task: str,
            inputs: List[str] = None,
            context: List[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
        # Input Message
        context_as_str = list_of_dict_to_markdown_table(context)
        main_prompt = self.prompt_templates["preprocess"]
        input_message = populate_template(
            main_prompt,
            variables={
                "task": task,
                "inputs": inputs,
                "context_as_str": context_as_str,
            }
        )
        input_message_lower = input_message.lower()
        # Hints
        hints = self.prompt_templates["main_hint"]
        filtered_hints = [
            hint for hint in hints
            if any(keyword.lower() in input_message_lower for keyword in hint.get("keyword", []))
        ]
        filtered_hints_str = f"""
        ## Hints
        
        {list_of_dict_to_markdown_table(filtered_hints)}
        """.strip()        
        model_input_messages = [
            {"role": "user", "content": [{"type": "text", "text": input_message}]},
            {"role": "user", "content": [{"type": "text", "text": filtered_hints_str}]},
        ]
        model_input_messages_str = "\n".join([msg["content"][0]["text"] for msg in model_input_messages])
        self.logger.log(text=model_input_messages_str, title=f"Augmented_LLM_Input({self.interface_id}/{self.name}):")

        previous_environment_errors: List[Dict[str, Any]] = []

        while True:
            model_input_messages_with_errors = copy.deepcopy(model_input_messages)

            if previous_environment_errors:
                error_str = "\n\n".join([
                    f"Error encountered:\n{err['error']}\nProblematic code:\n{err['code']}"
                    for err in previous_environment_errors
                ])
                model_input_messages_with_errors.append(
                    {"role": "system", "content": [{"type": "text", "text": error_str}]}
                )    
            try:
                code_output_message: ChatMessage = self.model(model_input_messages_with_errors)
            except Exception as e:
                raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
            observations = None
            try:
                code_action = self.executor_environment.parse_code_blobs(code_output_message.content)
            except Exception as e:
                error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                error_msg = self.executor_environment.parse_error_logs(error_msg)
                previous_environment_errors.append({"code": code_output_message.content, "error": error_msg})
                continue
            
            self.logger.log(text=code_action, title=f"Code Output ({self.interface_id}/{self.name}):")
            try:
                observations, _, _ = self.executor_environment(
                    code_action=code_action,
                    additional_variables={
                        "inputs": pd.DataFrame(context)
                    },
                    return_type="pandas.DataFrame"
                )
                self.logger.log(text=observations, title=f"Output from code execution: {len(observations)} characters")
            except Exception as e:
                error_msg = "Error in Code Execution: \n"
                if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                    error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
                error_msg += str(e)
                error_msg = self.executor_environment.parse_error_logs(error_msg)
                self.logger.log(text=f"Code: {code_action}\n\nError: {error_msg}", title="Error in Code Execution:")
                previous_environment_errors.append({"code": code_action, "error": error_msg})
                continue
            
            judge_input_message = {
                "role": MessageRole.USER, 
                "content": [{
                    "type": "text", 
                    "text": populate_template(
                        self.prompt_templates["judge"],
                        variables={
                            "task": model_input_messages_str,
                            "code": code_action,
                            "observations": observations,
                        }
                    )
                }]
            }
            self.logger.log(text=judge_input_message["content"][0]["text"], title=f"Judge Input ({self.interface_id}/{self.name}):")
            judge_output_message: ChatMessage = self.model([judge_input_message])

            self.judge_step = JudgeStep([judge_input_message], judge_output_message)
            self.memory.append_step(self.judge_step)

            decision = self.judge_step.to_decision()
            feedback = self.judge_step.get_feedback_content()

            self.logger.log(text=self.judge_step.model_output_message.content, title=f"Judge Output ({self.interface_id}/{self.name}):")
            self.logger.log(text=decision, title=f"Judge Decision ({self.interface_id}/{self.name}):")

            if decision == "approve":
                break
            else:
                previous_environment_errors = [{"code": code_action, "error": feedback}]

        return observations
    
    
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
        # TODO Setup task
        # TODO Setup inputs
        # TODO Setup context
        task: str = task.get("task")
        inputs: List[str] = task.get("inputs", None)
        context: List[Dict[str, Any]] = task.get("context", None)
        if not isinstance(task, str) or not isinstance(inputs, List[str]) or not isinstance(context, List[Dict[str, Any]]):
            raise ValueError("Task should be a string, inputs should be a list of strings, and context should be a list of dictionaries.")
        observations = None
        try:
            preprocessed_inputs = self.preprocess(task, inputs, context)
            outputs = self.main(preprocessed_inputs)
            # Return final answer
            observations = outputs
        except AgentError as e:
            self.logger.log(text=e.message, title="Error in Agent:")
            observations = pd.DataFrame([{"error": e.message}])
        
        return observations
                
    
    def get_final_answer(self) -> Any:
        code_output_message_content = self.state.get("final_answer", None)
        if code_output_message_content is None:
            raise ValueError("No final answer found in the state.")
        try:
            code_action = self.executor_environment.parse_code_blobs(code_output_message_content)
            self.logger.log(title="Code:", text=code_action)
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            error_msg = self.executor_environment.parse_error_logs(error_msg)
            raise AgentExecutionError(error_msg, self.logger) from e
        try:
            observations, _, _ = self.executor_environment(
                code_action=code_action,
                additional_variables={},
            )
            self.executor_environment.set_storage(next_step_id=0, code_action=code_action)
            self.logger.log(text=observations, title=f"Output from code execution: {len(observations)} characters")
        except Exception as e:
            error_msg = "Error in Code Execution: \n"
            if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
            error_msg += str(e)
            error_msg = self.executor_environment.parse_error_logs(error_msg)
            self.logger.log(text=f"Code: {code_action}\n\nError: {error_msg}", title="Error in Code Execution:")
            raise AgentExecutionError(error_msg, self.logger) from e
        
        self.state["current"] = observations
        
        return self.executor_environment.get_final_storage()
    
    def setup_environment(self):
        # Get class name from config
        interface_name = self.executor_environment_config["interface"]

        # Find all registered ExecutionEnvironment subclasses
        from agentcompany.extensions.environments.local_python_executor import LocalPythonInterpreter
        from agentcompany.extensions.environments.local_postgres_executor import LocalPostgresInterpreter
        from agentcompany.extensions.environments.local_tfserving_executor import LocalTfServingInterpreter
        
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
            
    
    def forward(self, task: str) -> Any:
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)