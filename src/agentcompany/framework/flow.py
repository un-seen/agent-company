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
from agentcompany.framework.ambient import AmbientPattern

logger = getLogger(__name__)

ActionType = Literal["final_answer", "skip", "execute", "environment"]

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
    if not 'known_variables' in global_state:
        global_state["known_variables"] = {}
    if not 'known_variables' in state:
        state["known_variables"] = {}
    if out_id and out_id.startswith("$"):
        out_id = out_id[1:]
        out_id = state[out_id]            
        state["known_variables"][out_id] = output
        global_state["known_variables"][out_id] = output
        if "final_answer" in state:
            state["final_answer"] = Template(state["final_answer"]).render(**state["known_variables"])
    if out_id:
        state[out_id] = output
    if output:
        state["current"] = output
        

class FlowPattern(AmbientPattern):
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
    output_type = "string"
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

        super().__init__()
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
        self.setup_variable_system()
        self.setup_logger_and_memory()
        # Context
        self.input_messages = None
        self.task = None
        # Vision
        self.plan_step = None


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
            variables = copy.deepcopy(self.state)
            variables["mcp_servers"] = self.mcp_servers
            prompt = populate_template(step, variables=variables)
            self.logger.log(text=f"Out={out} | Out_id={out_id}", title=f"Step {i} ({self.interface_id}/{self.name}):")
            if out == "one_to_many":
                output, _ = self._run_step(prompt, action_type, return_type, self.state)

                if not isinstance(output, list):
                    raise ValueError(f"Expected list output for 'one_to_many', got {type(output)}")
                
                if len(output) == 0:
                    raise ValueError("Empty output for 'one_to_many' step")
                
                next_steps = plan[i + 1:]
                for item in output:
                    local_state = copy.deepcopy(self.state)
                    local_state["mcp_servers"] = self.mcp_servers
                    set_state_out_id(self.state, local_state, out_id, item)
                    next_step_index = 0
                    previous_environment_errors = None
                    failures = 0
                    while next_step_index < len(next_steps):
                        next_step = next_steps[next_step_index]
                        next_step_out = next_step.get("out", "one_to_one")
                        if next_step_out != "one_to_one":
                            # TODO support other node outs like many_to_one and one_to_many
                            raise ValueError("After one_to_many, only one_to_one is allowed")
                        next_step_out_id = next_step.get("out_id")
                        next_step_action_type = next_step.get("action", "execute")
                        next_step_return_type = next_step.get("return_type", "string")
                        # Render the step
                        next_prompt = populate_template(next_step["step"], variables=local_state)
                        # Run the next step
                        next_output, previous_environment_errors = self._run_step(next_prompt, next_step_action_type, next_step_return_type, local_state, return_on_fail=True, previous_environment_errors=previous_environment_errors)
                        if next_output is None:
                            print(f"Step {i} failed with error: {previous_environment_errors}")
                            failures += 1
                            if failures > 3:
                                next_step_index += 1
                            else:
                                # Keep previous_environment_errors
                                next_step_index = 0
                                set_state_out_id(self.state, local_state, out_id, item)
                            continue
                        else:
                            previous_environment_errors = None
                            # Set output in local state out id
                            set_state_out_id(self.state, local_state, next_step_out_id, next_output)
                            next_step_index += 1

                break  # Exiting the loop as subsequent steps were processed already

            elif out == "one_to_one":
                output, _ = self._run_step(prompt, action_type, return_type, self.state)
                set_state_out_id(self.state, self.state, out_id, output)
            elif out == "many_to_one":
                if not isinstance(self.state["current"], list):
                    raise ValueError("Expected list in 'current' for 'many_to_one'")
                output, _ = self._run_step(prompt, action_type, return_type, self.state)
                set_state_out_id(self.state, self.state, out_id, output)
            i += 1
                
        
    def _run_step(self, prompt: str, action_type: ActionType, return_type: ReturnType, state: dict, return_on_fail = False, previous_environment_errors = None) -> Optional[Tuple[Any, List[Dict[str, Any]]]]:
        model_input_messages = [
            {"role": "system", "content": [{"type": "text", "text": self.description}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        previous_environment_errors: List[Dict[str, Any]] = previous_environment_errors or []

        if action_type == "environment":
            observations = call_method(self.executor_environment, prompt, state["current"])
        else:
                
            while True:
                model_input_messages_with_errors = copy.deepcopy(model_input_messages)
                if previous_environment_errors and len(previous_environment_errors) > 0:
                    error_str = "\n\n".join([
                        f"\n\n Please avoid these errors:\n\n{err['error']}"
                        for err in previous_environment_errors
                    ])
                    model_input_messages_with_errors.append(
                        {"role": "user", "content": [{"type": "text", "text": error_str}]}
                    )        
                model_input_messages_str = "\n".join([msg["content"][0]["text"] for msg in model_input_messages_with_errors])
                self.logger.log(text=model_input_messages_str, title=f"Flow_Pattern_Run_Step_LLM_Input({self.interface_id}/{self.name}):")
                try:
                    code_output_message: ChatMessage = self.model(model_input_messages_with_errors, return_type)
                except Exception as e:
                    raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
                observations = None
                code_action = None
                if action_type == "execute":
                    try:
                        code_action = self.executor_environment.parse_code_blob(code_output_message.content)
                    except Exception as e:
                        self.logger.log(text=f"Code: {code_output_message.content}\n\nError: {e}", title="Error in Code Parsing:")
                        error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                        error_msg = self.executor_environment.parse_error_logs(error_msg)
                        previous_environment_errors.append({"code": code_output_message.content, "error": error_msg})
                        continue
                    try:
                        observations, _, _ = self.executor_environment(
                            code_action=code_action,
                            additional_variables={
                                
                            },
                            return_type="string"
                        )
                    except Exception as e:
                        error_msg = "Error in Code Execution: \n"
                        if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                            error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
                        error_msg += str(e)
                        error_msg = self.executor_environment.parse_error_logs(error_msg)
                        self.logger.log(text=error_msg, title="CodeActionError:")
                        previous_environment_errors.append({"code": code_action, "error": error_msg})
                        continue
                elif action_type == "skip":
                    code_action = code_output_message.content
                    observations = code_output_message.content
                elif action_type == "final_answer":
                    code_action = code_output_message.content
                    observations = code_output_message.content
                    self.state["final_answer"] = observations                
                else:
                    raise ValueError(f"UnknownActionType({action_type})")

                self.logger.output(text=observations)
                self.logger.log(text=code_action, title=f"Flow_Pattern_Run_Step_LLM_Output({self.interface_id}/{self.name}):")
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
                self.logger.output(text=feedback)
                self.logger.log(text=self.judge_step.model_output_message.content, title=f"Judge Output ({self.interface_id}/{self.name}): {decision}")
                self.logger.action(text=code_action)

                if decision == "approve":
                    break
                elif return_on_fail:
                    previous_environment_errors = [{"code": code_action, "error": feedback}]
                    return None, previous_environment_errors
                else:
                    previous_environment_errors = [{"code": code_action, "error": feedback}]

        return observations, previous_environment_errors


    def get_final_answer(self) -> Any:
        code_action = self.state.get("final_answer", None)
        code_action = self.executor_environment.parse_code_blob(code_action)
        if code_action is None:
            raise ValueError("No final answer found in the state.")
        try:
            self.logger.log(text=code_action, title="CodeAction:")
            known_variables = self.state.get("known_variables", {})
            self.logger.log(text=known_variables, title="Known Variables:")
            observations, _, _ = self.executor_environment(
                code_action=code_action,
                additional_variables=known_variables,
            )
        except Exception as e:
            error_msg = "Error in Code Execution: \n"
            if hasattr(self.executor_environment, "state") and "_print_outputs" in self.executor_environment.state:
                error_msg += str(self.executor_environment.state["_print_outputs"]) + "\n\n"
            error_msg += str(e)
            error_msg = self.executor_environment.parse_error_logs(error_msg)
            self.logger.log(text=error_msg, title="CodeActionError:")
            raise AgentExecutionError(error_msg, self.logger) from e
        
        self.state["current"] = observations
        return observations
            
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
        self.task = task
        
        # Add Environment Variables
        if environment_variables is not None:
            self.state.update(environment_variables)
            
        if reset:
            self.memory.reset()
        self.memory.append_step(TaskStep(task=self.task))
        # Get Variables
        if self.variable_system:
            variable_table = self.variable_system_config["table"]
            variable_column_name = self.variable_system_config["column_name"]
            print(f"Task: {task} | Table: {variable_table} | Column: {variable_column_name}")
            variable_list = self.variable_system.get_variable_list(self.task, variable_table, variable_column_name)
            if variable_list is not None:
                variable_key, variable_data = variable_list
                self.state["known_entity"] = variable_key
                self.state["known_variables"] = variable_data
                self.state["variable_statement"] = []
                for key, value in variable_data.items():
                    self.state["variable_statement"].append(f"{key} = {value}\n")
                self.state["variable_statement"] = "\n".join(self.state["variable_statement"])
                
        # Execute Task
        observations = None
        try:
            # Execute vision
            self.executor_environment.state.update(self.state)
            self._execute_plan()
            observations = self.get_final_answer()
        except AgentError as e:
            self.logger.log(text=e.message, title="AgentError:")
            observations = e.message

        if self.variable_system and "known_variables" in self.state and "known_entity" in self.state:
            variable_table = self.variable_system_config["table"]
            variable_column_name = self.variable_system_config["column_name"]
            self.variable_system.set_variable_list(self.state["known_entity"], variable_table, variable_column_name, self.state["known_variables"])
        return observations        
        
    
    def forward(self, task: str) -> Any:
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)