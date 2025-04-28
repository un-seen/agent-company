
from typing import TypedDict
from jinja2 import StrictUndefined, Template    
from typing import Dict, Any, List

def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


class ExecutionEnvironmentConfig(TypedDict):
    """
    Configuration for the execution environment.
    """
    interface: str
    config: Dict[str, Any]


