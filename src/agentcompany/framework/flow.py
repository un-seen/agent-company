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
from agentcompany.llms.memory import VisionStep, JudgeStep, ValidateStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, FunctionCall, PlanningStepStatus
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
    
    

class FlowPattern(ModelContextProtocolImpl):
    """
    Agent class that takes as input a prompt and calls if required one or more agents to execute a vision.
    
    Args:
        name (`str`): Name of the agent.
        session_id (`str`): Session ID of the agent.
        interface_id (`str`): Interface ID of the agent.
        description (`str`): Description of the agent.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates (`PromptTemplates`, *optional*): Prompt templates for the agent.
        mcp_servers (`list`, *optional*): Managed agents that the agent can call.
    """
    
    description = "This is an agent implementing the flow design pattern."
    name = "FlowPattern"
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
        mcp_servers: List[ModelContextProtocolImpl],
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
        # MCP Servers
        self.setup_mcp_servers(mcp_servers)
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
        
    def setup_mcp_servers(self, mcp_servers: List[ModelContextProtocolImpl]):
        self.mcp_servers = {}
        if mcp_servers:
            assert all(server.name and server.description for server in mcp_servers), (
                "All managed agents need both a name and a description!"
            )
            self.mcp_servers = {server.name: server for server in mcp_servers}
            
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
        
        # Environment State
        self.state = copy.deepcopy(self.prompt_templates)
        self.state.pop("executor_environment", None)
        self.state.pop("judge", None)
        self.state.pop("plan", None)
        self.state.pop("hint", None)
        self.state["task"] = task
        
        # Add Environment Variables
        if environment_variables is not None:
            self.state.update(environment_variables)
            
        if reset:
            self.memory.reset()
        self.memory.append_step(TaskStep(task=self.task))
        # Execute Task
        observations = None
        try:
            # Execute vision
            self._execute_plan()
            # TODO add a status table in markdown
            status_table = self.get_status_table()
            self.logger.log(text=status_table, title=f"Final Vision Status ({self.interface_id}/{self.name}) :")
            # Return final answer
            observations = self.get_final_answer()
        except AgentError as e:
            self.logger.log(text=e.message, title="Error in Agent:")
            observations = pd.DataFrame([{"error": e.message}])
        
        return observations        
    
    
    # TODO add get status table as a markdown text
    def get_status_table(self):
        """
        Get the status table of the agent.
        """
        # TODO implement this method
        return ""
    
    
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
                return_type="string"
            )
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
        return observations
    
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
        
    def _execute_plan(self) -> None:
        plan: List[Node] = self.prompt_templates["plan"]
        state = copy.deepcopy(self.state)
        i = 0
        while i < len(plan):
            node = plan[i]
            step = node["step"]
            out = node.get("out", "one_to_one")
            out_id = node.get("out_id")
            action_type = node.get("action", "execute")
            return_type = node.get("return_type", "string")
            # Replace placeholders in the step with values from state
            from jinja2 import Template
            template: Template = Template(step)
            self.logger.log(title=f"Step {i} ({self.interface_id}/{self.name})", text=state)
            rendered_step = template.render(**state)
            self.logger.log(text=rendered_step, title=f"Step {i} ({self.interface_id}/{self.name}) Out-> {out}, Out_id-> {out_id}:")
            if out == "one_to_many":
                output = self._run_step(rendered_step, action_type, return_type)

                if not isinstance(output, list):
                    raise ValueError(f"Expected list output for 'one_to_many', got {type(output)}")

                state["current"] = output
                if out_id:
                    state[out_id] = output

                next_steps = plan[i + 1:]
                for item in output:
                    local_state = state.copy()
                    local_state["current"] = item

                    for next_step in next_steps:
                        next_step_out = next_step.get("out", "one_to_one")
                        if next_step_out != "one_to_one":
                            # TODO support other node outs like many_to_one and one_to_many
                            raise ValueError("After one_to_many, only one_to_one is allowed")
                        next_step_out_id = next_step.get("out_id")
                        next_step_action_type = next_step.get("action", "execute")
                        next_step_return_type = next_step.get("return_type", "string")
                        template_next: Template = Template(next_step["step"])
                        # Render the step
                        rendered_next_step = template_next.render(**local_state)
                        # Run the next step
                        next_output = self._run_step(rendered_next_step, next_step_action_type, next_step_return_type)
                        # Set output in local state out id
                        if next_step_out_id:
                            local_state[next_step_out_id] = next_output
                        local_state["current"] = next_output                           

                break  # Exiting the loop as subsequent steps were processed already

            elif out == "one_to_one":
                output = self._run_step(rendered_step, action_type, return_type)
                state["current"] = output
                if out_id:
                    state[out_id] = output
            elif out == "many_to_one":
                if not isinstance(state["current"], list):
                    raise ValueError("Expected list in 'current' for 'many_to_one'")
                output = self._run_step(rendered_step, action_type, return_type)
                state["current"] = output
                if out_id:
                    state[out_id] = output
            i += 1
            
            if i == len(plan):
                # End of plan request final answer from environment
                state["final_answer"] = state.get("current", None)
                self.state.update(state)
                break
    
    def _run_step(self, prompt: str, action_type: ActionType, return_type: ReturnType) -> None:
        hints = self.prompt_templates["hint"]
        prompt_lower = prompt.lower()    

        filtered_hints = [
            hint for hint in hints
            if any(keyword.lower() in prompt_lower for keyword in hint.get("keyword", []))
        ]
        filtered_hints_str = f"""
        ## Hints
        
        {list_of_dict_to_markdown_table(filtered_hints)}
        """.strip()
        system_prompt = self.description

        model_input_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": filtered_hints_str}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
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
                code_output_message: ChatMessage = self.model(model_input_messages_with_errors, return_type)
            except Exception as e:
                raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
            observations = None
            if return_type == "string":
                try:
                    code_action = self.executor_environment.parse_code_blobs(code_output_message.content)
                except Exception as e:
                    error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                    error_msg = self.executor_environment.parse_error_logs(error_msg)
                    previous_environment_errors.append({"code": code_output_message.content, "error": error_msg})
                    continue
            elif return_type == "list":
                code_action = code_output_message.content
            else:
                raise ValueError(f"Unknown return type: {return_type}")
            
            self.logger.log(text=code_action, title=f"Code Output ({self.interface_id}/{self.name}):")
            
            if action_type == "execute":
                try:
                    observations, _, _ = self.executor_environment(
                        code_action=code_action,
                        additional_variables={},
                        return_type="string"
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
            elif action_type == "skip":
                observations = code_output_message.content
            elif action_type == "final_answer":
                observations = code_output_message.content
                self.state["final_answer"] = observations                
            else:
                raise ValueError(f"Unknown action type: {action_type}")

            if not observations:
                previous_environment_errors.append({"code": code_action, "error": "There is no output for the code."})
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
                            "mcp_servers": self.mcp_servers,
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
    
    def forward(self, task: str) -> Any:
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)