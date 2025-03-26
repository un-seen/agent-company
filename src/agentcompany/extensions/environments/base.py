import abc
import re
from typing import Dict
from typing import Tuple, Callable, Any, TypedDict, Union, List
from agentcompany.mcp.base import ModelContextProtocolImpl

# TODO - Implement on top of the ModelContextProtocolImpl class

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

    state: Dict[str, Any]
    mcp_servers: Dict[str, ModelContextProtocolImpl]
    
    @abc.abstractmethod
    def __init__(self, mcp_servers: Dict[str, ModelContextProtocolImpl], **kwargs):
        raise NotImplementedError("ExecutionEnvironment not implemented.")
    
    @abc.abstractmethod
    def parse_code_blobs(self, code_blobs: str) -> str:
        raise NotImplementedError("parse_code_blobs not implemented.")
    
    def parse_error_logs(self, execution_logs: str) -> str:
        return execution_logs
        
    @abc.abstractmethod
    def attach_variables(self, variables: dict):
        raise NotImplementedError("attach_variables not implemented.")
    
    @abc.abstractmethod
    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        raise NotImplementedError("attach_mcp_servers not implemented.")
    
    @abc.abstractmethod
    def __call__(self, code_action: str, additional_variables: dict) -> Tuple[str, str, bool]:
        return super().__call__(code_action, additional_variables)
    
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