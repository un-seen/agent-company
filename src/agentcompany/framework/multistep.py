import inspect
import time
import re
from collections import deque
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import json
from agentcompany.llms.monitoring import (
    AgentLogger,
)
import textwrap
from redis import Redis
import os
from agentcompany.extensions.environments.base import ExecutionEnvironment, Observations
from agentcompany.llms.memory import ActionStep, JudgeStep, ValidateStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, FunctionCall, PlanningStepStatus
from agentcompany.driver.errors import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
)
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.extensions.pyfunc.base import FinalAnswerFunction
from agentcompany.mcp.utils import truncate_content
from agentcompany.llms.monitoring import LogLevel
from agentcompany.framework.prompt_template import PromptTemplates, EMPTY_PROMPT_TEMPLATES, populate_template
from agentcompany.llms.base import (
    ChatMessage,
    BaseLLM
)
from agentcompany.llms.utils import (
    MessageRole
)
from agentcompany.driver.xml import plan_xml_to_dict, plan_dict_to_xml, plan_dict_to_markdown_with_status, step_dict_to_xml, plan_dict_to_markdown_without_status, step_xml_to_dict


logger = getLogger(__name__)


class ReActPattern(ModelContextProtocolImpl):
    """
    Agent class that solves the given task step by step, using the ReAct design pattern:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

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
    
    description = "This is an agent implementing the ReAct design pattern."
    name = "ReActPattern"
    inputs = {"task": {
        "type": "string",
        "description": "The task for the agent to solve.",
    }}
    output_type = "string"
    interface_id: str = None
    executor_environment: ExecutionEnvironment = None

    def __init__(
        self,
        name: str,
        interface_id: str,
        description: str,
        model: BaseLLM,
        prompt_templates: PromptTemplates,
        mcp_servers: List[ModelContextProtocolImpl],
        step_callbacks: List[Callable],
        final_answer_checks: List[Callable],
        final_answer_call: ModelContextProtocolImpl = None,
        max_steps: int = 6,
    ):
        # Identifiers
        self.name = name
        self.description = description
        self.interface_id = interface_id
        # Logging
        verbosity_level: int = 1
        self.logger = AgentLogger(name, interface_id, level=verbosity_level, use_redis=True)
        # Storage Client
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
        # Prompt Templates
        self.prompt_templates = prompt_templates
        # LLM
        self.model = model
        # Planning
        planning_interval = 1
        self.planning_interval = planning_interval
        # Environment State
        self.state = {}
        # MCP Servers
        self.setup_mcp_servers(mcp_servers)
        self.final_answer_checks = final_answer_checks
        # Environment
        self.executor_environment_config = self.prompt_templates["executor_environment"]
        self.setup_environment()
        # System Prompt
        self.system_prompt = self.initialize_system_prompt()
        # Context
        self.input_messages = None
        self.task = None
        self.max_steps = max_steps
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        # Memory
        self.memory = AgentMemory(name, interface_id, self.system_prompt)
        super().__init__()
        # Clean Input Messages
        self.redis_client.delete(f"{self.interface_id}/{self.name}/input_messages")
        self.redis_client.delete(f"{self.interface_id}/{self.name}/plan")
        self.redis_client.delete(f"{self.interface_id}/{self.name}/fact")
        self.redis_client.delete(f"{self.interface_id}/{self.name}/ack")

    @property
    def logs(self):
        return [self.memory.system_prompt] + self.memory.steps

    def set_verbosity_level(self, level: int):
        self.logger.set_level(level)
    
    def write_memory_to_messages(
        self,
        summary_mode: Optional[bool] = False,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages
    
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
            self.redis_client.publish(self.interface_id, json.dumps({"role": self.name, "content": { "text": f"Using execution environment '{interface_name}'"}}))
        except KeyError:
            available = list(environment_classes.keys())
            raise ValueError(
                f"Unknown execution environment '{interface_name}'. "
                f"Available implementations: {available}"
            ) from None

        # Instantiate the chosen class
        self.executor_environment = environment_cls(
            self.mcp_servers,
            **self.executor_environment_config["config"]
        )
        self.executor_environment.attach_mcp_servers(self.mcp_servers)
        
    def setup_mcp_servers(self, mcp_servers: List[ModelContextProtocolImpl]):
        self.mcp_servers = {}
        if mcp_servers:
            assert all(server.name and server.description for server in mcp_servers), (
                "All managed agents need both a name and a description!"
            )
            self.mcp_servers = {server.name: server for server in mcp_servers}
            
    def initialize_system_prompt(self) -> str:
        variables={
            "mcp_servers": self.mcp_servers,
        }
        if "system_prompt_variables" in self.executor_environment_config:
            variables.update({
                variable: getattr(self.executor_environment, variable)
                for variable in self.executor_environment_config["system_prompt_variables"] if hasattr(self.executor_environment, variable)
            })
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables=variables,
        )
        return system_prompt
    

    def provide_final_answer(self, task: str, previous_observations: List[Observations]) -> str:
        """
        Provide the final answer to the task, based on the previous observations

        Args:
            task (`str`): Task to perform.
            previous_observations (`list[Observations]`, *optional*): Previous Observations.

        Returns:
            `str`: Final answer to the task.
        """
        # Execute code in multiple attempts     
        previous_environment_errors = []
        updated_task = task
        while True:
            # Check previous CoT
            if len(previous_environment_errors) > 0:
                variables = {
                    "previous_environment_errors": previous_environment_errors, # [ {"code": code_action, "error": error_msg} ]
                    "next_step": task,  
                    "max_task_length": 250
                }
                prompt_engineer_input_message = {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["prompt_engineer"], 
                                variables=variables
                            ),
                        }
                    ],
                }
                self.logger.log(text=prompt_engineer_input_message["content"][0]["text"], title="Prompt Engineer Input Message:")
                prompt_engineer_message: ChatMessage = self.model([prompt_engineer_input_message])
                updated_task = prompt_engineer_message.content
                self.logger.log(text=prompt_engineer_message.content, title="Prompt Engineer Output Message:")
            # Set Input Messages
            input_messages = []
            # Add System Prompt
            input_messages.extend(self.memory.system_prompt.to_messages(summary_mode=False))
            variables = {
                "task": self.task,
                "role": self.description,
                "observations": previous_observations,
            }
            final_code_message = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["final_answer"], 
                            variables=variables
                        ),
                    }
                ],
            }
            input_messages.extend([final_code_message])
            # Log Input Messages to LLM 
            input_messages_str = "\n".join([msg["content"][0]["text"] for msg in input_messages])
            self.logger.log(
                text=input_messages_str,
                title=f"Augmented_LLM_Input({self.interface_id}/{self.name}):"
            )
            # Execute LLM
            try:
                code_output_message: ChatMessage = self.model(
                    self.input_messages
                )
                self.logger.log(
                    text=code_output_message.content,
                    title=f"Augmented_LLM_Output({self.interface_id}/{self.name}):"
                )
            except Exception as e:
                raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
            
            # Parse code as per exection environment
            try:
                code_action = self.executor_environment.parse_code_blobs(code_output_message.content)
                self.logger.log(title="Code:", text=code_action)
            except Exception as e:
                error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                error_msg = self.executor_environment.parse_error_logs(error_msg)
                previous_environment_errors.append({"code": code_action, "error": error_msg, "prompt": updated_task})
                continue
            
            # Execute code in environment
            try:
                # Environment Code Compiles!
                observations, execution_logs, _ = self.executor_environment(code_action=code_action, additional_variables={})
                self.logger.log(text=observations, title=f"Output from code execution: {len(observations)} characters")
            except Exception as e:
                # Environment Code Compilation Error or Runtime Error!
                error_msg = "Error in Code Execution: \n"
                if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                    error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
                error_msg += str(e)
                error_msg = self.executor_environment.parse_error_logs(error_msg)
                previous_environment_errors.append({"code": code_action, "error": error_msg, "prompt": updated_task})
                continue
            
            if len(observations) == 0:
                previous_environment_errors.append({"code": code_action, "error": "There is no output for the code.", "prompt": updated_task})
                continue
            
            # Judge
            judge_input_message = {
                "role": MessageRole.USER, 
                "content": [
                    {
                        "type": "text", 
                        "text": populate_template(
                            self.prompt_templates["planning"]["judge"],
                            variables={
                                "task": updated_task,
                                "code": code_action,
                                "observations": observations,
                            }
                        )
                    }
                ]
            }
            self.logger.log(text=judge_input_message["content"][0]["text"], title="Judge Input:")
            judge_output_message: ChatMessage = self.model(
                [judge_input_message]
            )
            # Record Judge Step
            self.judge_step = JudgeStep([judge_input_message], judge_output_message)
            self.memory.append_step(self.judge_step)
            # Judge Decision and Guidance
            decision = self.judge_step.to_decision()
            guidance = self.judge_step.get_guidance_content()
            self.logger.log(text=self.judge_step.model_output_message.content, title="Judge Output:")
            self.logger.log(text=decision, title="Judge Decision:")
            # Set Judge Step Gate
            if decision == "approve":
                break
            elif decision == "reattempt" or decision == "step":
                previous_environment_errors = [{"code": code_action, "error": guidance, "prompt": updated_task}]
            else:
                raise AgentError(f"Unknown decision: {decision}", self.logger)
        
        return observations

    def execute_mcp_request(self, server_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_mcp_servers = {**self.mcp_servers}
        if server_name not in available_mcp_servers:
            error_msg = f"Unknown server {server_name}, should be instead one of {list(available_mcp_servers.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                observation = available_mcp_servers[server_name].__call__(arguments)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                observation = available_mcp_servers[server_name].__call__(**arguments)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg, self.logger)
            return observation
        except Exception as e:
            error_msg = (
                f"Error in calling mcp server: {e}\nYou should only ask this server with a correct request.\n"
                f"As a reminder, this server's description is the following:\n{available_mcp_servers[server_name]}"
            )
            raise AgentExecutionError(error_msg, self.logger)
    
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
        
    def run(
        self,
        task: str,
        reset: bool = True,
        images: Optional[List[str]] = None,
        environment_variables: Optional[Dict] = None,
    ) -> Union[str, None]:
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[str]`, *optional*): Paths to image(s).
            environment_variables (`dict`): Any environment variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task.

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
            self.executor_environment.attach_variables(environment_variables)
        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
        self.memory.append_step(TaskStep(task=self.task, task_images=images))
        # Execute Task
        final_answer = None
        try:
            # Make initial plan
            self._generate_initial_plan(task)
            # Execute plan
            previous_observations = self._execute_plan()
            # No more next steps after execute plan
            status_table = self.planning_step.get_markdown_table()
            self.logger.log(text=status_table, title="Final Plan Status:")
            # Return final answer
            final_answer = self.provide_final_answer(self.task, previous_observations)        
        except AgentError as e:
            self.logger.log(text=e.message, title="Error in Agent:")
            final_answer = e.message
        
        return final_answer        
        

    def _generate_initial_plan(self, task: str) -> None:
        # Empty Facts Initially
        if len(self.prompt_templates["planning"]["initial_facts"]) > 0:
            variables = {}
            if "initial_facts_variables" in self.executor_environment_config:
                variables.update({
                    variable: getattr(self.executor_environment, variable)
                    for variable in self.executor_environment_config["initial_facts_variables"] if hasattr(self.executor_environment, variable)
                })
            self.facts_message = ChatMessage(role=MessageRole.ASSISTANT, content=populate_template(
                self.prompt_templates["planning"]["initial_facts"],
                variables=variables,
            ))
        else:
            self.facts_message = ChatMessage(role=MessageRole.ASSISTANT, content="")
        # Initial Plan
        variables = {
            "role": self.description,
            "task": task,
            "facts": self.facts_message.content,
            "max_steps": self.max_steps,
        }
        if "initial_plan_variables" in self.executor_environment_config:
            variables.update({
                variable: getattr(self.executor_environment, variable)
                for variable in self.executor_environment_config["initial_plan_variables"] if hasattr(self.executor_environment, variable)
            })
        message_prompt_plan = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["initial_plan"],
                        variables=variables,
                    ),
                }
            ],
        }
        self.plan_message: ChatMessage = self.model([message_prompt_plan])
        self.logger.log(text=self.plan_message.content, title=f"Initial Plan Message Output ({self.interface_id}/{self.name}):")
        # Add to Memory
        self.planning_step = PlanningStep(facts=self.facts_message.content, plan=self.plan_message.content)
        self.memory.append_step(self.planning_step)
    
    def _update_plan_last_step(self, step_id: int, code_action: str, feedback: str):
        self.logger.log(title=f"Planning Step: {step_id}", text=feedback)
        next_step = self.planning_step.get_step(step_id)
        # Facts
        variables = {
            "next_step": next_step,
            "role": self.description,
            "facts": self.facts_message.content,
            "code": code_action,
            "feedback": feedback,            
        }
        if "update_facts_variables" in self.executor_environment_config:
            variables.update({
                variable: getattr(self.executor_environment, variable)
                for variable in self.executor_environment_config["update_facts_variables"] if hasattr(self.executor_environment, variable)
            })
        facts_update = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text", 
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_facts"],
                        variables=variables,
                    ),
                }
            ],
        }
        # Facts Message
        self.logger.log(text=facts_update["content"][0]["text"], title="Update Facts Input:")
        self.facts_message = self.model([facts_update])
        self.logger.log(text=self.facts_message.content, title="Update Facts Output:")
        # Plan
        plan_status_table = self.planning_step.get_markdown_table()
        variables = {
            "role": self.description,
            "task": self.task,
            "facts": self.facts_message.content,
            "plan_status_table": plan_status_table,
            "next_step": next_step,
            "code": code_action,
            "feedback": feedback,
        }
        update_plan = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_last_step"], 
                        variables=variables
                    ),
                }
            ],
        }
        plan_message: ChatMessage = self.model([update_plan])
        self.logger.log(text=plan_message.content, title="Updated Plan Message:")
        # Update Planning Step
        self.planning_step.update_step(step_id, plan_message.content)
        
    def _update_plan_next_step(self, step_id: int, previous_observations: List[Observations]) -> str:
        next_step = self.planning_step.get_step(step_id)
        variables = {
            "role": self.description,
            "next_step": next_step,
            "observations": previous_observations,
        }
        update_plan_next_step = {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_next_step"], 
                        variables=variables
                    ),
                }
            ],
        }
        next_step_plan_message: ChatMessage = self.model([update_plan_next_step])
        self.logger.log(text=next_step_plan_message, title="Updated Plan Message:")
        self.planning_step.update_step(step_id, next_step_plan_message.content)
    
    def _execute_plan(self) -> Union[None, List[Observations]]:
        """
        Execute the plan as per the ReAct framework: the agent thinks, acts, and observes the result.
        """
        # Check if plan is available
        if self.planning_step is None:
            raise AgentError("No planning is available to execute.", self.logger)
        # Execute code in multiple attempts     
        previous_observations = []
        previous_environment_errors = []
        while True:
            # Next Step Node
            next_step_id, next_step = self.planning_step.get_next_step()
            self.logger.log(text=f"{next_step_id}: {next_step}", title="Next Step")
            # Entry Gate Node
            previous_observations = self.executor_environment.get_previous_observations(next_step_id)
            if next_step is None or len(next_step) == 0:
                break
            # Create an updated next step node
            updated_next_step = next_step
            # if observations satisfy the task, return observations and continue to next step
            if isinstance(previous_observations, list) and len(previous_observations) > 0:
                validate_previous_approved_observations = self._validate_observations(next_step, previous_observations)
                if validate_previous_approved_observations == "approve":
                    self.planning_step.set_status(next_step_id, "approve")
                    continue
                elif validate_previous_approved_observations == "fail" or \
                        validate_previous_approved_observations == "rethink" or\
                            validate_previous_approved_observations == "step":
                    self._update_plan_next_step(next_step_id, previous_observations)
                else:
                    raise AgentError(f"Unknown validate decision: {validate_previous_approved_observations}", self.logger)
            # Check previous CoT
            if len(previous_environment_errors) > 0:
                variables = {
                    "previous_environment_errors": previous_environment_errors, # [ {"code": code_action, "error": error_msg} ]
                    "next_step": next_step,  
                    "mcp_servers": self.mcp_servers,
                    "max_task_length": 150,
                    "common_prompting_errors": self.prompt_templates["planning"]["common_prompting_errors"],
                }
                prompt_engineer_input_message = {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["prompt_engineer"], 
                                variables=variables
                            ),
                        }
                    ],
                }
                # Add all action steps to input messages
                # for step in self.memory.steps:
                #    if isinstance(step, ActionStep):
                #        self.input_messages.extend(step.to_messages())
                self.logger.log(text=prompt_engineer_input_message["content"][0]["text"], title="Prompt Engineer Input Message:")
                prompt_engineer_message: ChatMessage = self.model([prompt_engineer_input_message])
                updated_next_step = prompt_engineer_message.content
                self.logger.log(text=prompt_engineer_message.content, title="Prompt Engineer Output Message:")
            # Set Input Messages
            self.input_messages = []
            # Add System Prompt
            self.input_messages.extend(self.memory.system_prompt.to_messages(summary_mode=False))
            # Add prompt engineer
            self.input_messages.extend([{"role": MessageRole.USER, "content": [{"type": "text", "text": f"\n\n{updated_next_step}"}]}])
            # Log Input Messages to LLM 
            input_messages_str = "\n".join([msg["content"][0]["text"] for msg in self.input_messages])
            self.logger.log(
                text=input_messages_str,
                title=f"Augmented_LLM_Input({self.interface_id}/{self.name}):"
            )
            
            # Execute LLM
            try:
                code_output_message: ChatMessage = self.model(
                    self.input_messages
                )
                self.logger.log(
                    text=code_output_message.content,
                    title=f"Augmented_LLM_Output({self.interface_id}/{self.name}):"
                )
            except Exception as e:
                raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
            
            # Parse code as per exection environment
            try:
                code_action = self.executor_environment.parse_code_blobs(code_output_message.content)
                self.logger.log(title="Code:", text=code_action)
            except Exception as e:
                error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                error_msg = self.executor_environment.parse_error_logs(error_msg)
                previous_environment_errors.append({"code": code_action, "error": error_msg, "prompt": updated_next_step})
                continue
            
            # Execute code in environment
            try:
                # Environment Code Compiles!
                observations, execution_logs, _ = self.executor_environment(code_action=code_action, additional_variables={})
                self.logger.log(text=observations, title=f"Output from code execution: {len(observations)} characters")
            except Exception as e:
                # Environment Code Compilation Error or Runtime Error!
                error_msg = "Error in Code Execution: \n"
                if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                    error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
                error_msg += str(e)
                error_msg = self.executor_environment.parse_error_logs(error_msg)
                previous_environment_errors.append({"code": code_action, "error": error_msg, "prompt": updated_next_step})
                continue
            
            if len(observations) == 0:
                previous_environment_errors.append({"code": code_action, "error": "There is no output for the code.", "prompt": updated_next_step})
                continue
            
            # Judge
            judge_input_message = {
                "role": MessageRole.USER, 
                "content": [
                    {
                        "type": "text", 
                        "text": populate_template(
                            self.prompt_templates["planning"]["judge"],
                            variables={
                                "task": next_step,
                                "code": code_action,
                                "observations": observations,
                            }
                        )
                    }
                ]
            }
            self.logger.log(text=judge_input_message["content"][0]["text"], title="Judge Input:")
            judge_output_message: ChatMessage = self.model(
                [judge_input_message]
            )
            # Record Judge Step
            self.judge_step = JudgeStep([judge_input_message], judge_output_message)
            self.memory.append_step(self.judge_step)
            # Judge Decision and Guidance
            decision = self.judge_step.to_decision()
            guidance = self.judge_step.get_guidance_content()
            self.logger.log(text=self.judge_step.model_output_message.content, title="Judge Output:")
            self.logger.log(text=decision, title="Judge Decision:")
            # Set Status
            self.planning_step.set_status(next_step_id, decision)
            # Set Judge Step Gate
            if decision == "rethink":
                previous_environment_errors = []
                self._update_plan_last_step(next_step_id, self.judge_step.model_output_message.content)
            elif decision == "fail" or decision == "step":
                previous_environment_errors = [{"code": code_action, "error": guidance, "prompt": updated_next_step}]
            elif decision == "approve":
                previous_environment_errors = []
                self.executor_environment.save_observations(next_step_id, next_step, code_action, observations, guidance)
            else:
                raise AgentError(f"Unknown decision: {decision}", self.logger)
        
        return previous_observations
    
    def forward(self, task: str):
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)