
import yaml
import importlib.resources
from typing import Callable, List, Dict  
from agentcompany.driver.dict import merge_dicts
from agentcompany.framework.sequence import SequencePattern
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.llms.openai import OpenAIServerLLM
from pathlib import Path

def SequenceAgent(name: str, 
                    session_id: str,
                    interface_id: str, 
                    description: str, 
                    custom_yaml_path: str = None,
                    mcp_servers: List[ModelContextProtocolImpl] = []) -> SequencePattern:
    """
    Create a Sequence code agent.
    """
    default_yaml_path = importlib.resources.files("agentcompany.extensions.prompts.sequence").joinpath("default.yaml")
    default_prompt_templates: Dict = yaml.safe_load(default_yaml_path.read_text())
    updated_prompt_templates: Dict = dict()
    if custom_yaml_path:
        custom_yaml_path = Path(custom_yaml_path)
        file_reader =  open(custom_yaml_path, 'r')
        custom_prompt_templates: Dict = yaml.safe_load(file_reader)
        updated_prompt_templates = merge_dicts(updated_prompt_templates, custom_prompt_templates)
    prompt_templates = merge_dicts(default_prompt_templates, updated_prompt_templates)
    
    return SequencePattern(
        name=name, 
        session_id=session_id,
        interface_id=interface_id, 
        description=description, 
        model=OpenAIServerLLM("gpt-4o-mini"),
        prompt_templates=prompt_templates,
        mcp_servers=mcp_servers,
    )