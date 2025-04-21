
import yaml
import importlib.resources
from typing import Callable, List, Dict  
from agentcompany.driver.dict import merge_dicts
from agentcompany.framework.function import FunctionPattern
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.llms.openai import OpenAIServerLLM
from pathlib import Path

def FunctionAgent(name: str, 
                 session_id: str,
                 interface_id: str, 
                 description: str, 
                 mod_yaml_path: str = None,
                 custom_yaml_path: str = None,
                 mcp_servers: List[ModelContextProtocolImpl] = [],
                 model_id="gpt-4o-mini") -> FunctionPattern:
    """
    Create a Function agent.
    """
    default_yaml_path = importlib.resources.files("agentcompany.extensions.prompts.function").joinpath("default.yaml")
    default_prompt_templates: Dict = yaml.safe_load(default_yaml_path.read_text())
    updated_prompt_templates = default_prompt_templates
    # Mod
    if mod_yaml_path:
        mod_yaml_path = importlib.resources.files("agentcompany.extensions.prompts.flow").joinpath(f"{mod_yaml_path}.yaml")
        mod_yaml_path = Path(mod_yaml_path)
        file_reader =  open(mod_yaml_path, 'r')
        mod_prompt_templates: Dict = yaml.safe_load(file_reader)
        updated_prompt_templates = merge_dicts(updated_prompt_templates, mod_prompt_templates)
    # Custom
    if custom_yaml_path:
        custom_yaml_path = Path(custom_yaml_path)
        file_reader =  open(custom_yaml_path, 'r')
        custom_prompt_templates: Dict = yaml.safe_load(file_reader)
        updated_prompt_templates = merge_dicts(updated_prompt_templates, custom_prompt_templates)
    prompt_templates = merge_dicts(default_prompt_templates, updated_prompt_templates)
    
    return FunctionPattern(
        name=name, 
        session_id=session_id,
        interface_id=interface_id, 
        description=description, 
        model=OpenAIServerLLM(model_id=model_id),
        prompt_templates=prompt_templates,
    )