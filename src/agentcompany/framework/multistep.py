import pandas as pd
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from agentcompany.llms.monitoring import (
    AgentLogger,
)
from redis import Redis
import os
from agentcompany.extensions.environments.base import ExecutionEnvironment, Observations
from agentcompany.llms.memory import ActionStep, JudgeStep, ValidateStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, FunctionCall, PlanningStepStatus
from agentcompany.driver.errors import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
)
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



class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        initial_facts (`str`): Initial facts prompt.
        initial_plan (`str`): Initial plan prompt.
        update_facts_pre_messages (`str`): Update facts pre-messages prompt.
        update_facts_post_messages (`str`): Update facts post-messages prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_facts: str
    initial_plan: str
    update_facts_pre_messages: str
    update_facts_post_messages: str
    update_plan_pre_messages: str
    update_plan_post_messages: str
    
class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """
    system_prompt: str
    system_prompt_variables: List[str]
    planning: PlanningPromptTemplate
    executor_environment_config: ExecutionEnvironmentConfig
    

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
    inputs = "The task for the agent to solve in plain text english."
    output_type = "pandas.DataFrame"
    interface_id: str = None
    executor_environment: ExecutionEnvironment = None

    def __init__(
        self,
        name: str,
        session_id: str,
        interface_id: str,
        description: str,
        model: BaseLLM,
        prompt_templates: PromptTemplates,
        mcp_servers: List[ModelContextProtocolImpl],
        final_answer_checks: List[Callable],
        max_steps: int = 6,
    ):
        # Identifiers
        self.name = name
        self.interface_id = interface_id
        self.session_id = session_id
        # LLM
        self.model = model
        # Environment State
        self.state = {}
        # Prompt Templates
        self.prompt_templates = prompt_templates
        # Environment
        self.executor_environment_config = self.prompt_templates["executor_environment"]
        # MCP Servers
        self.setup_mcp_servers(mcp_servers)
        self.final_answer_checks = final_answer_checks
        self.setup_environment()
        # System Prompt
        self.system_prompt = self.initialize_system_prompt()
        # Generate Facts
        self._generate_initial_facts()
        self.description = description
        # Logging
        verbosity_level: int = 1
        self.logger = AgentLogger(name, interface_id, level=verbosity_level, use_redis=True)
        # Storage Client
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
        # Context
        self.input_messages = None
        self.task = None
        self.max_steps = max_steps
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
        except KeyError:
            available = list(environment_classes.keys())
            raise ValueError(
                f"Unknown execution environment '{interface_name}'. "
                f"Available implementations: {available}"
            ) from None

        # Instantiate the chosen class
        self.executor_environment = environment_cls(
            self.session_id,
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
        self.task = task
        if environment_variables is not None:
            self.state.update(environment_variables)
            self.executor_environment.attach_variables(environment_variables)
        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
        self.memory.append_step(TaskStep(task=self.task))
        # Execute Task
        final_answer = None
        try:
            # Make initial plan
            self._generate_initial_plan(task)
            # Execute plan
            self._execute_plan()
            # No more next steps after execute plan
            status_table = self.planning_step.get_markdown_table()
            self.logger.log(text=status_table, title=f"Final Plan Status ({self.interface_id}/{self.name}) :")
            # Return final answer
            # Bridge from current environment to python code is always pd.DataFrame
            final_answer = self.executor_environment.get_final_storage()
        except AgentError as e:
            self.logger.log(text=e.message, title="Error in Agent:")
            final_answer = pd.DataFrame([{"error": e.message}])
        
        return final_answer        
        
    def _generate_initial_facts(self) -> None:
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
            self.facts_message = ChatMessage(role=MessageRole.ASSISTANT, content="\n\n")
            
        if len(self.mcp_servers) > 0:
            self.facts_message.content += "You can take into consideration knowledge of the following functions:\n\n"
            for server in self.mcp_servers:
                if isinstance(self.mcp_servers[server], ReActPattern):
                    mcp_server: ReActPattern = self.mcp_servers[server]
                    self.facts_message.content += f"""
                        {mcp_server.name} 
                        
                        inputs: {mcp_server.inputs}
                        returns: {mcp_server.output_type}
                        
                        {mcp_server.facts_message.content}
                        
                        {"\n\n".join([f"Task: {p["Task"]} \n Resolution: {p["Resolution"]}" for p in (mcp_server.prompt_templates["planning"]["common_prompting_errors"] or [])])}""".strip()
                    
        # Initial Plan
        variables = {
            "role": self.description,
            "task": task,
            "facts": self.facts_message.content,
            "max_steps": self.max_steps,
            "common_prompting_errors": self.prompt_templates["planning"]["common_prompting_errors"] or [],
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
        self.logger.log(text=message_prompt_plan["content"][0]["text"], title=f"Initial Plan Message Input ({self.interface_id}/{self.name}):")
        self.plan_message: ChatMessage = self.model([message_prompt_plan])
        self.logger.log(text=self.plan_message.content, title=f"Initial Plan Message Output ({self.interface_id}/{self.name}):")
        # Add to Memory
        self.planning_step = PlanningStep(facts=self.facts_message.content, plan=self.plan_message.content)
        # TODO validate the step in each plan
        # Because the step updates are always at a plan level
        # The approach of each step can be changed when executing
        # But the key result is always the same and it must be in sync
        # with other steps. Therefore the steps must be validated.
        # Put a while loop even if one step is improper regenerate the planning step
        self.memory.append_step(self.planning_step)
    
    def _update_plan_last_step(self, step_id: int, code_action: str, feedback: str) -> str:
        self.logger.log(title=f"Planning Step: {step_id}", text=feedback)
        next_step = self.planning_step.get_step(step_id)
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
        self.logger.log(text=plan_message.content, title="Updated Plan Last Step:")
        # Update Planning Step
        self.planning_step.update_step(step_id, plan_message.content)
        return plan_message.content
        
    def _update_plan_facts(self, observations: List[Observations]):
        if observations is None:
            self.logger.log(title="No observations to update facts.")
            return
        # Facts
        variables = {
            "role": self.description,
            "facts": self.facts_message.content,
            "observations": observations,            
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
        self.facts_message = self.model([facts_update])
        
    def _update_plan_next_step(self, step_id: int) -> str:
        next_step = self.planning_step.get_step(step_id)
        storage_data = self.get_storage_data(step_id)
        variables = {
            "role": self.description,
            "next_step": next_step,
            "storage_data": storage_data,
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
        self.logger.log(text=update_plan_next_step["content"][0]["text"], title=f"Update Plan Next Step Input ({self.interface_id}/{self.name}):")
        next_step_plan_message: ChatMessage = self.model([update_plan_next_step])
        self.logger.log(text=next_step_plan_message.content, title=f"Updated Plan Next Step Output ({self.interface_id}/{self.name}):")
        self.planning_step.update_step(step_id, next_step_plan_message.content)
        return next_step_plan_message.content
    
    def get_storage_data(self, step_id: int) -> List:
        storage_data = []
        for prev_step_id in range(0, step_id):
            prev_step = self.planning_step.get_original_step(prev_step_id)
            prev_step_storage = self.executor_environment.get_storage(prev_step_id)
            storage_data.append({
                "step": prev_step,
                "storage": prev_step_storage,
            })
        return storage_data
    
    def _execute_plan(self) -> None:
        """
        Execute the plan as per the ReAct framework: the agent thinks, acts, and observes the result.
        """
        # Check if plan is available
        if self.planning_step is None:
            raise AgentError("No planning is available to execute.", self.logger)
        # Execute code in multiple attempts     
        next_step_id = -1
        previous_environment_errors: List[EnvironmentError] = []
        while True:
            # Next Step Node
            next_step_id, next_step = self.planning_step.get_next_step()
            self.logger.log(text=f"{next_step_id}: {next_step}", title="Next Step")
            if next_step is None or len(next_step) == 0:
                break
            # Create an updated next step node
            updated_next_step = next_step
            # Update next step taking into account the previous results
            # With no environment errors till now
            if next_step_id > 0 and len(previous_environment_errors) == 0:
                self._update_plan_next_step(next_step_id)
                # previous_environment_errors = []
                updated_next_step = self.planning_step.get_step(next_step_id)
            # Check previous CoT    
            if len(previous_environment_errors) > 0:
                variables = {
                    "role": self.description,
                    "previous_environment_errors": previous_environment_errors, 
                    "next_step": next_step,  
                    "mcp_servers": self.mcp_servers,
                    "max_task_length": 150,
                    "common_prompting_errors": self.prompt_templates["planning"]["common_prompting_errors"] or [],
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
                previous_environment_errors.append({"code": code_action, "error": error_msg, "task": updated_next_step})
                continue
            
            # Execute code in environment
            try:
                # Environment Code Compiles!
                observations, _, _ = self.executor_environment(code_action=code_action, additional_variables={})
                self.logger.log(text=observations, title=f"Output from code execution: {len(observations)} characters")
            except Exception as e:
                # Environment Code Compilation Error or Runtime Error!
                error_msg = "Error in Code Execution: \n"
                if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                    error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
                error_msg += str(e)
                error_msg = self.executor_environment.parse_error_logs(error_msg)
                self.logger.log(text=f"Code: {code_action} \n\n Error: {error_msg}", title="Error in Code Execution:")
                previous_environment_errors.append({"code": code_action, "error": error_msg, "task": updated_next_step})
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
                                "task": updated_next_step,
                                "code": code_action,
                                "mcp_servers": self.mcp_servers,
                                "storage_data": self.get_storage_data(next_step_id),
                                "observations": observations,
                            }
                        )
                    }
                ]
            }
            self.logger.log(text=judge_input_message["content"][0]["text"], title=f"Judge Input ({self.interface_id}/{self.name}) :")
            judge_output_message: ChatMessage = self.model(
                [judge_input_message]
            )
            # Record Judge Step
            self.judge_step = JudgeStep([judge_input_message], judge_output_message)
            self.memory.append_step(self.judge_step)
            # Judge Decision and Feedback
            decision = self.judge_step.to_decision()
            feedback = self.judge_step.get_feedback_content()
            self.logger.log(text=self.judge_step.model_output_message.content, title=f"Judge Output ({self.interface_id}/{self.name}) :")
            self.logger.log(text=decision, title=f"Judge Decision ({self.interface_id}/{self.name}) :")
            # Set Status
            self.planning_step.set_status(next_step_id, decision)
            # Set Judge Step Gate
            if decision == "rethink":
                previous_environment_errors: List[EnvironmentError] = []
                self._update_plan_last_step(next_step_id, code_action, self.judge_step.model_output_message.content)
            elif decision == "fail" or decision == "step":
                previous_environment_errors: List[EnvironmentError] = [{"code": code_action, "error": feedback, "prompt": updated_next_step}]
            elif decision == "approve":
                previous_environment_errors: List[EnvironmentError] = []
                self.executor_environment.save_observations(next_step_id, next_step, code_action, observations, feedback)
                self.executor_environment.set_storage(next_step_id, code_action)
                self.logger.log(text=self.executor_environment.get_storage(next_step_id), title=f"Step Storage {next_step_id} ({self.interface_id}/{self.name}):")
                self._update_plan_facts(self.executor_environment.get_previous_observations(next_step_id))
            else:
                raise AgentError(f"Unknown decision: {decision}", self.logger)
        
    
    def forward(self, task: str) -> Any:
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)