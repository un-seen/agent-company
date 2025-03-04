
import yaml
import importlib.resources
from typing import Callable, List
from agentcompany.framework.multistep import ReActPattern

def PythonCodeAgent(name: str, 
                    interface_id: str, 
                    description: str, 
                    mcp_servers: List[Callable] = [],
                    step_callbacks: List[Callable] = [],
                    final_answer_checks: List[Callable] = [], 
                    final_answer_call: Callable = print) -> ReActPattern:
    """
    Create a Python code agent.
    """
    prompt_templates = yaml.safe_load(
        importlib.resources.files("agentcompany.lib.prompts").joinpath("python.yaml").read_text()
    )
    return ReActPattern(
        name=name, 
        interface_id=interface_id, 
        description=description, 
        model="model",
        prompt_templates=prompt_templates,
        mcp_servers=mcp_servers,
        step_callbacks=step_callbacks,
        final_answer_checks=final_answer_checks,
        final_answer_call=final_answer_call,
        max_steps=4,
    )