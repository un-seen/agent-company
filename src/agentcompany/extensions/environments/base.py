import abc
import re
from typing import Dict
from typing import Tuple, Callable, Any
from agentcompany.mcp.base import ModelContextProtocolImpl

# TODO - Implement on top of the ModelContextProtocolImpl class
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
    def __call__(self, code_action: str, additional_variables: dict) -> Tuple[Any, str, bool]:
        return super().__call__(code_action, additional_variables)