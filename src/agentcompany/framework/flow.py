import pandas as pd
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
from agentcompany.llms.monitoring import (
    AgentLogger,
)
from jinja2 import Template
import json
import copy
from typing_extensions import Literal
from agentcompany.llms.base import (
    ReturnType
)
from agentcompany.extensions.environments.postgres_sql_executor import PostgresSqlInterpreter
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

type ActionType = Literal["final_answer", "skip", "execute", "environment"]

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
    
    

def call_method(obj: Any, method_name: str, *args, **kwargs) -> Any:
    """
    Dynamically call `obj.method_name(*args, **kwargs)` and return its result.
    
    Raises:
      AttributeError   – if `obj` has no attribute `method_name`
      TypeError        – if the attribute isn’t callable
      Any exception from the underlying method call will propagate.
    """
    # 1. Get the attribute
    try:
        method = getattr(obj, method_name)
    except AttributeError:
        raise AttributeError(f"'{type(obj).__name__}' object has no method '{method_name}'")
    
    # 2. Ensure it’s callable
    if not callable(method):
        raise TypeError(f"Attribute '{method_name}' of '{type(obj).__name__}' is not callable")
    
    # 3. Call with provided args/kwargs
    return method(*args, **kwargs)


def set_state_out_id(global_state: dict, state: dict, out_id: str, output: Any) -> None:
    """
    Set the out_id in the state.
    """
    if out_id.startswith("$"):
        out_id = out_id[1:]
        out_id = state[out_id]
        if not 'known_variables' in state:
            state["known_variables"] = {}
        if not 'known_variables' in global_state:
            global_state["known_variables"] = {}
            
        state["known_variables"][out_id] = output
        global_state["known_variables"][out_id] = output
        if "final_answer" in state:
            state["final_answer"] = Template(state["final_answer"]).render(**state["known_variables"])
    state[out_id] = output
    state["current"] = output
        

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
        self.description = description
        self.interface_id = interface_id
        self.session_id = session_id
        # LLM
        self.model = model
        # Prompt Templates
        self.prompt_templates = prompt_templates
        # State
        self.state = {}
        # MCP Servers
        self.setup_mcp_servers(mcp_servers)
        # Environment
        self.executor_environment_config = self.prompt_templates["executor_environment"]
        self.setup_environment()
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
        self.state.update(self.executor_environment.state)
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
            self.executor_environment.state.update(self.state)
            self._execute_plan()
            # TODO add a status table in markdown
            status_table = self.get_status_table()
            self.logger.log(text=status_table, title=f"Final Status ({self.interface_id}/{self.name}) :")
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
        code_action = self.state.get("final_answer", None)
        if code_action is None:
            raise ValueError("No final answer found in the state.")
        try:
            self.logger.log(text=f"{code_action}", title="Final Answer:")
            observations, _, _ = self.executor_environment(
                code_action=code_action,
                additional_variables=self.state["known_variables"],
            )
            self.logger.log(text=observations, title="Final Answer Observations:")
        except Exception as e:
            error_msg = "Error in Code Execution: \n"
            if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
            error_msg += str(e)
            error_msg = self.executor_environment.parse_error_logs(error_msg)
            self.logger.log(text=f"Code: {code_action}\n\nError: {error_msg}", title="Error in code execution (get_final_answer):")
            raise AgentExecutionError(error_msg, self.logger) from e
        
        self.state["current"] = observations
        return observations
    
    def setup_environment(self):
        # Get class name from config
        interface_name = self.executor_environment_config["interface"]

        # Find all registered ExecutionEnvironment subclasses
        from agentcompany.extensions.environments.jupyter_python_executor import JupyterPythonInterpreter
        from agentcompany.extensions.environments.postgres_sql_executor import PostgresSqlInterpreter
        from agentcompany.extensions.environments.b2_text_executor import B2TextInterpreter
        
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
    
    def setup_hint(self):
        step_lower = self.state["task"].lower()    
        prompt_hints = self.prompt_templates.get("hint", [])
        prompt_hints = [
            hint for hint in prompt_hints
            if any(keyword.lower() in step_lower for keyword in hint.get("keyword", []))
        ]
        prompt_hints = list_of_dict_to_markdown_table(prompt_hints)
        if len(prompt_hints) > 0:
            self.state["hint"] = f"""
            ## Hints
            
            {prompt_hints}
            """.strip()
        else:
            self.state["hint"] = ""
            
    def _execute_plan(self) -> None:
        plan: List[Node] = self.prompt_templates["plan"]
        i = 0
        self.setup_hint()
        while i < len(plan):
            node = plan[i]
            step = node["step"]
            out = node.get("out", "one_to_one")
            out_id = node.get("out_id")
            action_type = node.get("action", "execute")
            return_type = node.get("return_type", "string")
            # Replace placeholders in the step with values from state
            template: Template = Template(step)
            rendered_step = template.render(**self.state)
            self.logger.log(text=f"Out={out} | Out_id={out_id}", title=f"Step {i} ({self.interface_id}/{self.name}):")
            if out == "one_to_many":
                output = self._run_step(rendered_step, action_type, return_type, self.state)

                if not isinstance(output, list):
                    raise ValueError(f"Expected list output for 'one_to_many', got {type(output)}")
                
                if len(output) == 0:
                    raise ValueError("Empty output for 'one_to_many' step")
                
                next_steps = plan[i + 1:]
                for item in output:
                    local_state = copy.deepcopy(self.state)
                    set_state_out_id(self.state, local_state, out_id, item)
                    next_step_index = 0
                    while next_step_index < len(next_steps):
                        next_step = next_steps[next_step_index]
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
                        next_output = self._run_step(rendered_next_step, next_step_action_type, next_step_return_type, local_state, return_on_fail=True)
                        if next_output is None:
                            next_step_index = 0
                            set_state_out_id(self.state, local_state, out_id, item)
                            continue
                        else:
                            # Set output in local state out id
                            set_state_out_id(self.state, local_state, next_step_out_id, next_output)
                            next_step_index += 1

                break  # Exiting the loop as subsequent steps were processed already

            elif out == "one_to_one":
                output = self._run_step(rendered_step, action_type, return_type, self.state)
                set_state_out_id(self.state, self.state, out_id, output)
            elif out == "many_to_one":
                if not isinstance(self.state["current"], list):
                    raise ValueError("Expected list in 'current' for 'many_to_one'")
                output = self._run_step(rendered_step, action_type, return_type, self.state)
                set_state_out_id(self.state, self.state, out_id, output)
            i += 1
                
        
    def _run_step(self, prompt: str, action_type: ActionType, return_type: ReturnType, state: dict, return_on_fail = False) -> Optional[str]:
        model_input_messages = [
            {"role": "system", "content": [{"type": "text", "text": self.description}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        previous_environment_errors: List[Dict[str, Any]] = []

        while True:
            
            # Check if it is a deterministic action because itself and it's output are both defined in the environment
            if action_type == "environment":
                print(f"State: {state}")
                print(f"Current: {state['current']}")
                observations = call_method(self.executor_environment, prompt, state["current"])
                print(f"Observations: {observations}")
                break
            
            model_input_messages_with_errors = copy.deepcopy(model_input_messages)

            if previous_environment_errors:
                error_str = "\n\n".join([
                    f"Please avoid this error:\n\n{err['error']}"
                    for err in previous_environment_errors
                ])
                model_input_messages_with_errors.append(
                    {"role": "user", "content": [{"type": "text", "text": error_str}]}
                )        
            model_input_messages_str = "\n".join([msg["content"][0]["text"] for msg in model_input_messages_with_errors])
            self.logger.log(text=model_input_messages_str, title=f"Augmented_LLM_Input({self.interface_id}/{self.name}):")
            try:
                code_output_message: ChatMessage = self.model(model_input_messages_with_errors, return_type)
            except Exception as e:
                raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
            observations = None
            if return_type == "string":
                try:
                    code_action = self.executor_environment.parse_code_blob(code_output_message.content)
                except Exception as e:
                    self.logger.log(text=f"Code: {code_output_message.content}\n\nError: {e}", title="Error in code parsing:")
                    error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                    error_msg = self.executor_environment.parse_error_logs(error_msg)
                    previous_environment_errors.append({"code": code_output_message.content, "error": error_msg})
                    continue
            elif return_type == "list":
                code_action = code_output_message.content
            else:
                raise ValueError(f"Unknown return type: {return_type}")
        
            self.logger.log(text=f"```{self.executor_environment.language} \n {code_action} \n```", title=f"Code Output ({self.interface_id}/{self.name}):")
        
            if action_type == "execute":
                try:
                    observations, _, _ = self.executor_environment(
                        code_action=code_action,
                        additional_variables={
                            
                        },
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
                state["final_answer"] = observations                
            else:
                raise ValueError(f"Unknown action type: {action_type}")

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
            elif return_on_fail:
                return None
            else:
                previous_environment_errors = [{"code": code_action, "error": feedback}]

        return observations
    
    def forward(self, task: str) -> Any:
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)