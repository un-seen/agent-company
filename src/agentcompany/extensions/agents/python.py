
import yaml
import importlib.resources
from typing import Callable, List, Dict  
from agentcompany.framework.multistep import ReActPattern
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.llms.openai import OpenAIServerLLM

def PythonCodeAgent(name: str, 
                    interface_id: str, 
                    description: str, 
                    mcp_servers: List[Callable] = [],
                    step_callbacks: List[Callable] = [],
                    final_answer_checks: List[Callable] = [], 
                    final_answer_call: ModelContextProtocolImpl = None) -> ReActPattern:
    """
    Create a Python code agent.
    """
    default_yaml_path = importlib.resources.files("agentcompany.extensions.prompts").joinpath("default.yaml")
    default_prompt_templates: Dict = yaml.safe_load(default_yaml_path.read_text())
    agent_yaml_path = importlib.resources.files("agentcompany.extensions.prompts").joinpath("python.yaml")
    prompt_templates: Dict = yaml.safe_load(agent_yaml_path.read_text())
    prompt_templates.update(default_prompt_templates)
    
    return ReActPattern(
        name=name, 
        interface_id=interface_id, 
        description=description, 
        model=OpenAIServerLLM("gpt-4o-mini"),
        prompt_templates=prompt_templates,
        mcp_servers=mcp_servers,
        step_callbacks=step_callbacks,
        final_answer_checks=final_answer_checks,
        final_answer_call=final_answer_call,
        max_steps=4,
    )