# TODO implement this to run "table" generation using brave api and jina api for agent quant
# TODO combine corner with duckdb agent to create and analyze datasets in agentquant
# TODO make a company of corner agent and duckdb agent to create and analyze datasets in agentquant
from logging import getLogger
from typing import Dict, List, Optional
import json
from agentcompany.llms.monitoring import (
    AgentLogger,
)
from redis import Redis
import os
from agentcompany.extensions.environments.base import ExecutionEnvironment
from agentcompany.llms.memory import AgentMemory
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.llms.base import (
    BaseLLM,
    ChatMessage,
)
from agentcompany.llms.base import (
    ChatMessage,
)
from agentcompany.llms.utils import (
    MessageRole
)
from agentcompany.framework.base import FrameworkPattern

logger = getLogger(__name__)


class FunctionRunner:
    def __init__(self, func):
        self.func = func

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    

class ShotPattern(ModelContextProtocolImpl, FrameworkPattern):
    """
    Agent class that solves the given task in one shot. 

    Args:
        name (`str`): Name of the agent.
        interface_id (`str`): Interface ID of the agent.
        description (`str`): Description of the agent.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
    """
    
    description = "This is an agent implementing the shot design pattern."
    name = "ShotPattern"
    inputs = {"task": {
        "type": "string",
        "description": "The task for the agent to solve.",
    }}
    output_type = "string"
    interface_id: str = None
    executor_environment: ExecutionEnvironment = None

    def __init__(
        self,
        name: str,
        interface_id: str,
        description: str,
        model: BaseLLM,
    ):
        # Identifiers
        self.name = name
        self.description = description
        self.interface_id = interface_id
        # Logging
        verbosity_level: int = 1
        self.logger = AgentLogger(name, interface_id, level=verbosity_level, use_redis=True)
        # Storage Client
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
        # LLM
        self.model = model
        # Environment State
        self.state = {}
        # Context
        self.input_messages = None
        self.task = None
        # Memory
        self.memory = AgentMemory(name, interface_id, "ShotPatternHasNoSystemPrompt")
                 
    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[str]] = None,
        environment_variables: Optional[Dict] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[str]`, *optional*): Paths to image(s).
            environment_variables (`dict`): Any environment variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task.

        Example:
        ```py
        from agentcompany.runtime.multistep import MultiStepAgent
        agent = MultiStepAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        self.task = task
        messages = [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": task,
                    }
                ],
            },
        ]
        try:
            chat_message: ChatMessage = self.model(messages)
            final_answer = chat_message.content
        except Exception as e:
            final_answer = f"Error in generating final LLM output:\n{e}"
        self.redis_client.publish(self.interface_id, json.dumps({"final_answer": final_answer}))
        return final_answer
    
    
    def forward(self, task: str):
        """
        MCPContextProtocolImpl forward method.
        """
        return self.run(task)