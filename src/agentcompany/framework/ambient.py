import os
from redis import Redis
from typing import Any, Dict, List, Union
from agentcompany.llms.monitoring import AgentLogger
from agentcompany.llms.memory import AgentMemory
from agentcompany.extensions.environments.base import ExecutionEnvironment
from agentcompany.driver.errors import AgentExecutionError
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.driver.markdown import list_of_dict_to_markdown_table

class AmbientPattern(ModelContextProtocolImpl):
    """
    AmbientPattern is a pattern that allows an agent to interact with the environment.
    """
    redis_client: Redis
    name: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])

    @property
    def logs(self):
        return [self.memory.system_prompt] + self.memory.steps

    def set_verbosity_level(self, level: int):
        self.logger.set_level(level)
    
    def write_memory_to_messages(
        self,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = []
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages())
        return messages

    def setup_environment(self):
        # Get class name from config
        interface_name = self.executor_environment_config["interface"]

        # Find all registered ExecutionEnvironment subclasses
        from agentcompany.extensions.environments.jupyter_python_executor import JupyterPythonInterpreter
        from agentcompany.extensions.environments.postgres_sql_executor import PostgresSqlInterpreter
        from agentcompany.extensions.environments.b2_text_executor import B2TextInterpreter
        
        environment_classes = {cls.__name__: cls for cls in ExecutionEnvironment.__subclasses__()}        
        try:
            environment_cls = environment_classes[interface_name]
        except KeyError:
            available = list(environment_classes.keys())
            raise ValueError(
                f"Unknown execution environment '{interface_name}'. "
                f"Available implementations: {available}"
            ) from None

        # Instantiate the chosen class
        self.executor_environment = environment_cls(
            self.session_id,
            self.mcp_servers,
            **self.executor_environment_config["config"]
        )
        self.executor_environment.attach_mcp_servers(self.mcp_servers)
    
    def setup_hint(self):
        step_lower = self.state["task"].lower()    
        prompt_hints = self.prompt_templates.get("hint", [])
        prompt_hints = [
            hint for hint in prompt_hints
            if any(keyword.lower() in step_lower for keyword in hint.get("keyword", []))
        ]
        if len(prompt_hints) > 0:
            prompt_hints = list_of_dict_to_markdown_table(prompt_hints)
            self.state["hint"] = f"""
            ## Hints
            
            {prompt_hints}
            """.strip()
        else:
            self.state["hint"] = ""
    
    def setup_variable_system(self):
        # Get class name from config
        self.variable_system = None
        variable_system_config = self.prompt_templates.get("variable_system", None)
        if variable_system_config is None:
            return
        interface_name = variable_system_config["interface"]
        if interface_name != "PostgresSqlInterpreter":
            raise ValueError(f"Unknown variable system '{interface_name}'.")
        # Instantiate the chosen class
        self.variable_system = PostgresSqlInterpreter(
            self.session_id,
            self.mcp_servers,
            **variable_system_config["config"]
        )
        variable_system_config.pop("config")
        self.variable_system_config = variable_system_config
    
    def setup_logger_and_memory(self):
        # Redis Client
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
        # Logging
        verbosity_level: int = 1
        self.logger = AgentLogger(self.name, self.interface_id, level=verbosity_level, use_redis=True)
        # Memory
        self.memory = AgentMemory(self.name, self.interface_id)
    
    def setup_mcp_servers(self, mcp_servers: List[ModelContextProtocolImpl]):
        self.mcp_servers = {}
        if mcp_servers:
            assert all(server.name and server.description for server in mcp_servers), (
                "All managed agents need both a name and a description!"
            )
            self.mcp_servers = {server.name: server for server in mcp_servers}


    def execute_mcp_request(self, server_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_mcp_servers = {**self.mcp_servers}
        if server_name not in available_mcp_servers:
            error_msg = f"Unknown server {server_name}, should be instead one of {list(available_mcp_servers.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                observation = available_mcp_servers[server_name].__call__(arguments)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                observation = available_mcp_servers[server_name].__call__(**arguments)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg, self.logger)
            return observation
        except Exception as e:
            error_msg = (
                f"Error in calling mcp server: {e}\nYou should only ask this server with a correct request.\n"
                f"As a reminder, this server's description is the following:\n{available_mcp_servers[server_name]}"
            )
            raise AgentExecutionError(error_msg, self.logger)