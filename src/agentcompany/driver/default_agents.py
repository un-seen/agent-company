from agentcompany.driver.agents import MultiStepAgent
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

class SupervisorAgent(MultiStepAgent):
    """
    In this agent, the LLM receives the final summary of the work done by a Manager
    then it will return a review of the work done by the Manager.

    Args:
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        system_prompt (`str`, *optional*): System prompt that will be used to generate the agent's actions.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        company_name: str,
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        system_prompt: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        max_print_outputs_length: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = MANAGER_SYSTEM_PROMPT
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.tools = []
        # TODO Add default agents
        managed_agents = kwargs.pop("managed_agents", None)
        all_managed_agent: bool = all([isinstance(agent, ManagedAgent) for agent in managed_agents])
        if managed_agents is None:
            raise ValueError("You need to provide managed agents to the ManagerAgent.")
        if not all_managed_agent:
            raise ValueError("All agents in managed_agents should be of type ManagedAgent.")
        
        super().__init__(
            tools=self.tools,
            model=model,
            system_prompt=system_prompt,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        if "*" in self.additional_authorized_imports:
            # self.logger.log(
            #     "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
            #     0,
            # )
            pass
        
        all_tools = {**self.tools, **self.managed_agents}
        self.python_executor = LocalPythonInterpreter(
            self.additional_authorized_imports,
            all_tools,
            max_print_outputs_length=max_print_outputs_length,
        )
        from redis import Redis
        import os
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
        self.company_name = company_name
        
    def initialize_system_prompt(self):
        self.system_prompt = super().initialize_system_prompt()
        self.system_prompt = self.system_prompt.replace(
            "{{authorized_imports}}",            
            "You can import from any package you want."
            if "*" in self.authorized_imports
            else str(self.authorized_imports),
        )
        return self.system_prompt

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages.copy()

        # Add new step in logs
        log_entry.model_input_messages = memory_messages.copy()
        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            chat_message: ChatMessage = self.model(
                self.input_messages,
                stop_sequences=["<end_code>", "Observation:"],
                **additional_args,
            )
            log_entry.model_output_message = chat_message
            model_output = chat_message.content
            log_entry.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        self.logger.log(
            Group(
                Rule(
                    "[italic]Output message of the LLM:",
                    align="left",
                    style="orange",
                ),
                Syntax(
                    model_output,
                    lexer="markdown",
                    theme="github-dark",
                    word_wrap=True,
                ),
            ),
            level=LogLevel.DEBUG,
        )

        # Parse
        try:
            code_action = fix_final_answer_code(parse_code_blobs(model_output))
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        log_entry.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        # Execute
        self.logger.log(
            Panel(
                Syntax(
                    code_action,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]Executing this code:",
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=LogLevel.INFO,
        )
        observation = ""
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(
                code_action,
                self.state,
            )
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation += "Execution logs:\n" + execution_logs
        except Exception as e:
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        log_entry.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        log_entry.action_output = output
        
        if is_final_answer:
            import json
            output_str = json.dumps({"answer": output, "agent": self.name})
            self.redis_client.publish(self.company_name, output_str)
        return output if is_final_answer else None
    
    

class SupervisorAgent(AgentCompanyAgent):
    # TODO provide method implementation for tools, environment and action space, to run Reason/Plan, Action, and Observation.
    # via a templatable prompt.
    name = "user_input"
    description = "This tool is used to get user input."
    inputs = {"prompt": {"type": "string", "description": "The prompt to show to the user."}}
    output_type = "string"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from redis import Redis
        import os
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
    def forward(self, prompt: str) -> str:
        import random
        random_seed = random.randint(0, 1000000)
        user_input_key = f"user_input:{random_seed}"
        self.redis_client.publish("final_answer", user_input_key)
        while True:
            user_input = self.redis_client.get(user_input_key)
            if user_input is not None:
                break
        return user_input



TOOL_MAPPING = {
    tool_class.name: tool_class
    for tool_class in [
        SupervisorAgent
    ]
}


__all__ = [
    "SupervisorAgent",
    "BashAgent",
    "SQLAgent",
    "PytorchAgent"
]
