import pandas as pd
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union
from agentcompany.llms.monitoring import (
    AgentLogger,
)
from jinja2 import Template
import copy
from typing_extensions import Literal
from agentcompany.llms.base import (
    ReturnType
)
from redis import Redis
import os
from agentcompany.extensions.environments.base import ExecutionEnvironment, Observations
from agentcompany.llms.memory import JudgeStep, AgentMemory, TaskStep
from agentcompany.driver.errors import (
    AgentError,
    AgentGenerationError,
)
from pydantic import BaseModel
from agentcompany.llms.base import AugmentedLLM
from agentcompany.driver.markdown import list_of_dict_to_markdown_table
from agentcompany.mcp.base import ModelContextProtocolImpl
from typing import TypedDict
from agentcompany.framework.prompt_template import ExecutionEnvironmentConfig, populate_template
from agentcompany.framework.ambient import AmbientPattern
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
    
    
class Variable(BaseModel):
    """
    Variable for the Agent Flow
    """
    name: str
    description: str

class VariableList(BaseModel):  
    """
    Variable list for the Agent Flow
    """
    variable_list: List[Variable]
    
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
      TypeError        – if the attribute isn't callable
      Any exception from the underlying method call will propagate.
    """
    # 1. Get the attribute
    try:
        method = getattr(obj, method_name)
    except AttributeError:
        raise AttributeError(f"'{type(obj).__name__}' object has no method '{method_name}'")
    
    # 2. Ensure it's callable
    if not callable(method):
        raise TypeError(f"Attribute '{method_name}' of '{type(obj).__name__}' is not callable")
    
    # 3. Call with provided args/kwargs
    return method(*args, **kwargs)


def set_state_out_id(global_state: dict, state: dict, out_id: str, output: Any) -> None:
    """
    Set the out_id in the state.
    """
    if not 'known_variables' in state:
        state["known_variables"] = {}
    if not 'known_variables' in global_state:
        global_state["known_variables"] = {}
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
        

class GraphPattern(ModelContextProtocolImpl):
    """
    Agent class that implements a computation pattern for tasks requiring multiple outputs on an input list.
    It creates a causal graph by backsolving from the root node or criteria created from the task using LLM completions,
    and then uses the causal graph to create output for each item in the list solving the task.
    
    Args:
        name (`str`): Name of the agent.
        session_id (`str`): Session ID of the agent.
        interface_id (`str`): Interface ID of the agent.
        description (`str`): Description of the agent.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates (`PromptTemplates`, *optional*): Prompt templates for the agent.
        mcp_servers (`list`, *optional*): Managed agents that the agent can call.
    """
    
    description = "This is an agent implementing the graph computation pattern."
    name = "GraphPattern"
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
        # Graph
        self.causal_graph = None
        self.root_node = None
        # Memory
        self.memory = AgentMemory(name, interface_id)
        super().__init__()

            
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
            observations = self.state["results"]
        except AgentError as e:
            self.logger.log(text=e.message, title="AgentError:")
            observations = pd.DataFrame([{"error": e.message}])
        
        return observations        

            
    def _build_causal_graph(self, task: str, input_list: List[dict]) -> List[Variable]:
        """
        Build a causal graph from the task by identifying root nodes and dependencies.
        Uses LLM to identify criteria and relationships between items in the input list.
        """
        # TODO: Implement logic to create causal graph
        # backsolved from the criteria as the terminal node
        # use the list of inputs, hint, previous reports etc
        # to create the causal graph
        # You parse the criteria into causal variables and a terminal node which is the criteria
        # and then you create a graph from the criteria to the input list via the causal variables
        # Generate criteria and relationships using LLM
        data_input_list = []
        for item in input_list:
            data = item.get("data", None)
            if data is None:
                raise ValueError("No data found in input list")
            data["_id"] = item.get("_id", None)
            data_input_list.append(data)
        data_input_list_as_markdown = list_of_dict_to_markdown_table(data_input_list)
        criteria_prompt = populate_template(
            self.prompt_templates["criteria"],
            variables={
                "task": task,
                "input_list_as_markdown": data_input_list_as_markdown
            }
        )
        self.logger.log(text=criteria_prompt, title="Criteria Prompt:")
        input_messages = [
            {"role": "system", "content": self.description},
            {"role": "user", "content": criteria_prompt}
        ]
        criteria_output = self.model(input_messages)
        criteria_output = criteria_output.content
        print(f"Criteria Output: {criteria_output}")
        # Parse criteria and build causal graph
        variables_prompt = populate_template(
            self.prompt_templates["variable"],
            variables={
                "task": self.task,
                "criteria": criteria_output,
                "input_list_as_markdown": data_input_list_as_markdown
            }
        )
        input_messages = [
            {"role": "system", "content": self.description},
            {"role": "user", "content": variables_prompt}
        ]
        client: AugmentedLLM = self.model
        variable_output: VariableList = client.structured_output(input_messages, output_schema=VariableList)
        print(f"Variable Output: {variable_output}")
        return variable_output.variable_list

    def _identify_root_node(self, graph: Dict) -> str:
        """
        Identify the root node in the causal graph.
        """
        # TODO: Implement root node identification logic
        return ""
    
    def _traverse_graph(self, node: str, item: Any) -> Any:
        """
        Traverse the causal graph starting from the given node for the current item.
        """
        # TODO given the item traverse the graph to get a value for the root node or the criteria node
        # then perhaps if it is ranking then do bubble sort or merge sort to create
        # a rankng based on the input graph 
        # Get node computation template
        node_template = self.prompt_templates["nodes"].get(node, {})
        if not node_template:
            raise ValueError(f"No template found for node: {node}")

        # Compute node output
        prompt = populate_template(
            node_template["step"],
            variables={
                "item": item,
                "node": node,
                "mcp_servers": self.mcp_servers,
                "state": self.state,
            }
        )

        output, _ = self._run_step(
            prompt,
            node_template.get("action", "execute"),
            node_template.get("return_type", "string"),
            self.state
        )

        # Process child nodes if any
        child_nodes = self.causal_graph.get(node, {}).get("children", [])
        child_outputs = {}
        for child in child_nodes:
            child_outputs[child] = self._traverse_graph(child, item)

        # Combine outputs based on node type
        return self._combine_outputs(output, child_outputs, node)

    def _combine_outputs(self, node_output: Any, child_outputs: Dict[str, Any], node: str) -> Any:
        """
        Combine the current node's output with its child nodes' outputs based on the node type.
        """
        # TODO: Implement output combination logic based on node type
        return node_output

    def _run_step(self, prompt: str, action_type: ActionType, return_type: ReturnType, state: dict, return_on_fail = False, previous_environment_errors = None) -> Optional[Tuple[Any, List[Dict[str, Any]]]]:
        model_input_messages = [
            {"role": "system", "content": [{"type": "text", "text": self.description}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        previous_environment_errors: List[Dict[str, Any]] = previous_environment_errors or []

        while True:
            
            # Check if it is a deterministic action because itself and it's output are both defined in the environment
            if action_type == "environment":
                observations = call_method(self.executor_environment, prompt, state["current"])
                break
            
            model_input_messages_with_errors = copy.deepcopy(model_input_messages)

            if previous_environment_errors:
                error_str = "\n\n".join([
                    f"\n\n Please avoid these errors:\n\n{err['error']}"
                    for err in previous_environment_errors
                ])
                model_input_messages_with_errors.append(
                    {"role": "user", "content": [{"type": "text", "text": error_str}]}
                )        
            model_input_messages_str = "\n".join([msg["content"][0]["text"] for msg in model_input_messages_with_errors])
            self.logger.log(text=model_input_messages_str, title=f"LLM_Input({self.interface_id}/{self.name}):")
            try:
                code_output_message: ChatMessage = self.model(model_input_messages_with_errors, return_type)
            except Exception as e:
                raise AgentGenerationError(f"Error in running llm:\n{e}", self.logger) from e
            observations = None
            if return_type == "string":
                try:
                    code_action = self.executor_environment.parse_code_blob(code_output_message.content)
                except Exception as e:
                    self.logger.log(text=f"Code: {code_output_message.content}\n\nError: {e}", title="Error in Code Parsing:")
                    error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
                    error_msg = self.executor_environment.parse_error_logs(error_msg)
                    previous_environment_errors.append({"code": code_output_message.content, "error": error_msg})
                    continue
            elif return_type == "list":
                code_action = code_output_message.content
            else:
                raise ValueError(f"Unknown return type: {return_type}")
        
            self.logger.log(code=code_action, title=f"LLM_Output({self.interface_id}/{self.name}):")
        
            if action_type == "execute":
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

            self.logger.log(text=self.judge_step.model_output_message.content, title=f"Judge Output ({self.interface_id}/{self.name}): {decision}")

            if decision == "approve":
                break
            elif return_on_fail:
                previous_environment_errors = [{"code": code_action, "error": feedback}]
                return None, previous_environment_errors
            else:
                previous_environment_errors = [{"code": code_action, "error": feedback}]

        return observations, previous_environment_errors

    def _execute_plan(self) -> None:
        """
        Execute the computation plan using the causal graph.
        For each item in the input list, traverse the graph and compute the required outputs.
        """
        # TODO set input list in state
        # TODO now set the variable value by asking the b2 text interpreter 
        # create a dict with the variable name and value
        # Set current item in state
        # self.state["current"] = item
        # # Traverse graph and compute outputs
        # output = self._traverse_graph(self.root_node, item)
        # results.append(output)
        input_code = self.prompt_templates.get("input", None)
        if input_code is None:
            raise ValueError("No input list provided in prompt templates")
        input_list, _, _ = self.executor_environment(code_action=input_code, additional_variables={})
        variables_list = self._build_causal_graph(self.state["task"], input_list)
        self.logger.log(text=list_of_dict_to_markdown_table([{"name": v.name, "description": v.description} for v in variables_list]), title="Variable List:")
        node_list = []
        for node in input_list:            
            _id = node.get("_id", None)
            if _id is None:
                raise ValueError("No _id found in input list")
            data = node.get("data", None)
            if data is None:
                raise ValueError("No data found in input list")
            
            data_as_markdown = list_of_dict_to_markdown_table([data])
            global_context = f"{_id} has these properties:\n {data_as_markdown}\n"
            print(f"Calculating Rank Vector for Node: {_id}")
            self.logger.log(text=global_context, title=f"Global Context for {_id}:")
            for variable in variables_list:
                # TODo fix variable
                local_context = f"{global_context} \n {variable.name}: {variable.description} \n" 
                question = f"What is the value of {variable.name} for {_id} ?"
                variable_value = self.executor_environment.web_qa(question, local_context)
                print(f"Question: {question} | Answer: {variable_value}")
                data[variable.name] = variable_value
            data["_id"] = _id
            node_list.append(data)
        # Set final results
        self.state["results"] = node_list
        # TODO Create a matrix
        # apply reranking
        # return the ranked list 
        # persist in postgres
        print(list_of_dict_to_markdown_table(node_list))
        raise ValueError("Stop here for now")
    
    def forward(self, task: str) -> Any:
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)