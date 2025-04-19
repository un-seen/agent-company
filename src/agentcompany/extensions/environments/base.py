import abc
import pandas as pd
from typing import Dict
from typing import Tuple, Callable, Any, TypedDict, Union, List
from agentcompany.mcp.base import ModelContextProtocolImpl

class Observations(TypedDict):
    step_id: int
    step: str
    code_action: str
    observations: str
    feedback: str
    
class EnvironmentError(TypedDict):
    code: str
    error: str
    task: str
    
class ExecutionEnvironment(abc.ABC):

    language: str 
    state: Dict[str, Any]
    storage: Dict[int, Dict[str, Any]] = {}
    session_id: str
    mcp_servers: Dict[str, ModelContextProtocolImpl]
    static_tools: Dict[str, Callable] = {}
    
    def __init__(self, session_id: str, mcp_servers: Dict[str, ModelContextProtocolImpl], **kwargs):
        self.session_id = session_id
        self.mcp_servers = mcp_servers
    
    @abc.abstractmethod
    def parse_code_blob(self, code_blob: str) -> str:
        raise NotImplementedError("parse_code_blobs not implemented.")
    
    @abc.abstractmethod
    def parse_function(self, code_blob: str) -> Dict[str, Callable]:
        raise NotImplementedError("parse_function not implemented.")
    
    def parse_error_logs(self, execution_logs: str) -> str:
        return execution_logs
        
    @abc.abstractmethod
    def attach_variables(self, variables: dict):
        raise NotImplementedError("attach_variables not implemented.")
    
    @abc.abstractmethod
    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        raise NotImplementedError("attach_mcp_servers not implemented.")
    
    @abc.abstractmethod
    def __call__(self, code_action: str, additional_variables: dict, return_type: str = "string") -> Tuple[str, str, bool]:
        return super().__call__(code_action, additional_variables)
    
    def set_storage(self, next_step_id: int, code_action: str, observations: List[Dict[str, Any]] = None):
        raise NotImplementedError("save_in_memory not implemented.")
    
    def get_storage_id(self, next_step_id: int) -> str:
        raise NotImplementedError("get_previous_memory_prompt not implemented.")
    
    def get_storage(self, storage_id: str) -> str:
        raise NotImplementedError("get_memory not implemented.")
    
    def save_observations(self, next_step_id: int, next_step: str, code_action: str, observations: str, feedback: str) -> Observations:
        if "observations" not in self.state:
            self.state["observations"] = {}
        step_observations: Observations = {
            "step_id": next_step_id,
            "step": next_step,
            "code_action": code_action,
            "observations": observations,
            "feedback": feedback
        }
        self.state["observations"][next_step_id] = step_observations
        return step_observations
    
    def get_previous_observations(self, next_step_id: int) -> Union[List[Observations], None]:
        if "observations" not in self.state:
            return None
        previous_observations = []
        for i in range(next_step_id):
            if i not in self.state["observations"]:
                break
            previous_observations.append(self.state["observations"][i])
        return previous_observations
    
    def get_final_storage(self) -> pd.DataFrame:
        raise NotImplementedError("get_final_storage not implemented.")