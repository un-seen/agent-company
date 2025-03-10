import inspect
import time
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
from agentcompany.extensions.environments.base import ExecutionEnvironment
from agentcompany.llms.memory import ActionStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, FunctionCall
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
from agentcompany.framework.base import FrameworkPattern

logger = getLogger(__name__)





class ReActPattern(ModelContextProtocolImpl, FrameworkPattern):
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
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
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
        self.mcp_servers["final_answer"] = final_answer_call or FinalAnswerFunction
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
            self.redis_client.publish(self.interface_id, json.dumps({"role": self.name, "text": f"Using execution environment '{interface_name}'"}))
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
            "mcp_servers": {
                server_name: server
                for server_name, server in self.mcp_servers.items()
            },
        }
        variables.update({
            variable: getattr(self.executor_environment, variable)
            for variable in self.executor_environment_config["system_prompt_variables"]  if hasattr(self.executor_environment, variable)
        })
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables=variables,
        )
        return system_prompt
        
    def extract_action(self, model_output: str, split_token: str) -> Tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: Optional[list[str]]) -> str:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        messages = [{"role": MessageRole.SYSTEM, "content": []}]
        messages[0]["content"] = [
            {
                "type": "text",
                "text": "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
            }
        ]
        # Add Images If Necessary
        if images:
            messages[0]["content"].append({"type": "image"})
        messages += self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": f"Based on the above, please provide an answer to the following user request:\n{task}",
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = self.model(messages)
            return chat_message.content
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"

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
    
    def _validate_final_answer(self, final_answer: Any):
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _execute_step(self, task: str, action_step: ActionStep) -> Union[None, Any]:
        if self.planning_interval is not None and self.step_number % self.planning_interval == 1:
            self._planning_step(task, is_first_step=(self.step_number == 1), step=self.step_number)
        final_answer = self.step(action_step)
        if final_answer is not None and self.final_answer_checks:
            self._validate_final_answer(final_answer)
        return final_answer
    
    def _finalize_step(self, action_step: ActionStep, step_start_time: float):
        action_step.end_time = time.time()
        action_step.duration = action_step.end_time - step_start_time
        self.memory.steps.append(action_step)
        for callback in self.step_callbacks:
            # For compatibility with old callbacks that don't take the agent as an argument
            callback(action_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                action_step, agent=self
            )
        self.step_number += 1
    
    def _handle_max_steps_reached(self, task: str, images: List[str], step_start_time: float) -> ActionStep:
        error_message = "Reached max steps."
        final_answer = self.provide_final_answer(task, images)
        final_action_step = ActionStep(
            step_number=self.step_number, error=AgentMaxStepsError(error_message, self.logger)
        )
        final_action_step = ActionStep(error=AgentMaxStepsError(error_message, self.logger))
        final_action_step.action_output = final_answer
        final_action_step.end_time = time.time()
        final_action_step.duration = final_action_step.end_time - step_start_time
        self.memory.steps.append(final_action_step)
        for callback in self.step_callbacks:
            # For compatibility with old callbacks that don't take the agent as an argument
            if len(inspect.signature(callback).parameters) == 1:
                callback(final_action_step)
            else:
                callback(final_action_step, agent=self)
        return final_action_step
    
    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[str]] = None,
        environment_variables: Optional[Dict] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
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
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))
        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run(task=self.task, images=images, max_steps=max_steps)
        # Outputs are returned only at the end as a string. We only look at the last step
        return deque(self._run(task=self.task, images=images, max_steps=max_steps), maxlen=1,)[0]
    
    def _run(self, task: str, images: List[str] | None = None, max_steps: int = None) -> Generator[ActionStep, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        final_answer = None
        self.step_number = 0
        max_steps = max_steps if max_steps is not None else self.max_steps
        while final_answer is None and self.step_number < max_steps:
            step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            try:
                # Run one step!
                final_answer = self._execute_step(task, action_step)
            except AgentError as e:
                action_step.error = e
            finally:
                self._finalize_step(action_step, step_start_time)
                yield action_step
                
        if final_answer is None and self.step_number == self.max_steps:
            final_action_step = self._handle_max_steps_reached(task, images, step_start_time)
            final_answer = final_action_step.action_output
            yield final_action_step

        yield final_answer

    def _planning_step(self, task, is_first_step: bool, step: int) -> None:
        input_messages, facts_message, plan_message = (
            self._generate_initial_plan(task) if is_first_step else self._generate_updated_plan(task, step)
        )
        self._record_planning_step(input_messages, facts_message, plan_message, is_first_step)
    
    def _generate_initial_plan(self, task: str) -> Tuple[ChatMessage, ChatMessage]:
        input_messages = [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["initial_facts"], variables={"task": task}
                        ),
                    }
                ],
            },
        ]
        facts_message = self.model(input_messages)

        message_prompt_plan = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["initial_plan"],
                        variables={
                            "task": task,
                            "mcp_servers": self.mcp_servers,
                            "answer_facts": facts_message.content,
                        },
                    ),
                }
            ],
        }
        plan_message = self.model([message_prompt_plan], stop_sequences=["<end_plan>"])
        return input_messages, facts_message, plan_message
    
    def _generate_updated_plan(self, task: str, step: int) -> Tuple[ChatMessage, ChatMessage]:

        update_plan_pre = {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                    ),
                }
            ],
        }
        # Do not take the system prompt message from the memory
        # summary_mode=False: Do not take previous plan steps to avoid influencing the new plan
        memory_messages = self.write_memory_to_messages()[1:]
        facts_update_pre = {
            "role": MessageRole.SYSTEM,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_pre_messages"]}],
        }
        facts_update_post = {
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": self.prompt_templates["planning"]["update_facts_post_messages"]}],
        }
        input_messages = [facts_update_pre] + memory_messages + [facts_update_post]
        facts_message = self.model(input_messages)
        update_plan_post = {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": populate_template(
                        self.prompt_templates["planning"]["update_plan_post_messages"],
                        variables={
                            "task": task,
                            "mcp_servers": self.mcp_servers,
                            "facts_update": facts_message.content,
                            "remaining_steps": (self.max_steps - step),
                        },
                    ),
                }
            ],
        }
        plan_message = self.model(
            [update_plan_pre] + memory_messages + [update_plan_post], stop_sequences=["<end_plan>"]
        )
        return input_messages, facts_message, plan_message

    def _record_planning_step(
        self, input_messages: list, facts_message: ChatMessage, plan_message: ChatMessage, is_first_step: bool
    ) -> None:
        if is_first_step:
            facts = textwrap.dedent(f"""Here are the facts that I know so far:\n```\n{facts_message.content}\n```""")
            plan = textwrap.dedent(
                f"""Here is the plan of action that I will follow to solve the task:\n```\n{plan_message.content}\n```"""
            )
        else:
            facts = textwrap.dedent(
                f"""Here is the updated list of the facts that I know:\n```\n{facts_message.content}\n```"""
            )
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere is my new/updated plan of action to solve the task:\n```\n{plan_message.content}\n```"""
            )
        self.memory.steps.append(
            PlanningStep(
                model_input_messages=input_messages,
                facts=facts,
                plan=plan,
                model_output_message_plan=plan_message,
                model_output_message_facts=facts_message,
            )
        )

    def step(self, action_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages.copy()

        # Add new step in logs
        action_step.model_input_messages = memory_messages.copy()
        try:
            chat_message: ChatMessage = self.model(
                self.input_messages,
                stop_sequences=["<end_code>", "Observation:"],
            )
            action_step.model_output_message = chat_message
            model_output = chat_message.content
            action_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e
        
        self.logger.log(
            text=model_output,
            title="Output message of the LLM:",
            level=LogLevel.INFO,
        )

        # TODO Parse as per exection environment
        try:
            code_action = self.executor_environment.parse_code_blobs(model_output)
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        action_step.function_calls = [
            FunctionCall(
                name=self.executor_environment_config["interface"],
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        self.logger.log(title="Executing parsed code:", text=code_action, level=LogLevel.INFO)
        is_final_answer = False
        
        try:
            # TODO pass additional variables instead of {}
            output, execution_logs, is_final_answer = self.executor_environment.__call__(code_action, {})
            observation = execution_logs
        except Exception as e:
            if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                execution_logs = str(self.executor_environment.state["_print_outputs"])
                action_step.observations = execution_logs
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        action_step.observations = observation
        self.logger.log(text=truncated_output, title="Output from code snippet:" if is_final_answer else "Final Answer from code snippet:", level=LogLevel.INFO)
        action_step.action_output = output
        if is_final_answer:
            self.redis_client.publish(self.interface_id, json.dumps({"role": self.name, "text": truncated_output, "title": "Final Answer"}))
            return output
        return None
    
    def forward(self, task: str):
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)