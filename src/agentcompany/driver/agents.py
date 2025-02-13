import inspect
import time
from collections import deque
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
import json

from .monitoring import (
    AgentLogger,
    LogLevel,
)

from redis import Redis
import os
            
from .memory import ActionStep, AgentMemory, PlanningStep, SystemPromptStep, TaskStep, ToolCall
from .types import AgentImage, handle_agent_output_types
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    parse_code_blobs,
    parse_json_tool_call,
    truncate_content,
)
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    LocalPythonInterpreter,
    fix_final_answer_code,
)
from .models import (
    ChatMessage,
    MessageRole,
)
from .prompts import (
    PYTHON_CODE_SYSTEM_PROMPT,
    MANAGED_AGENT_PROMPT,
    SUPERVISOR_AGENT_PROMPT,
    MANAGER_SYSTEM_PROMPT,
    PLAN_UPDATE_FINAL_PLAN_REDACTION,
    SYSTEM_PROMPT_FACTS,
    SYSTEM_PROMPT_FACTS_UPDATE,
    SYSTEM_PROMPT_PLAN,
    SYSTEM_PROMPT_PLAN_UPDATE,
    TOOL_CALLING_SYSTEM_PROMPT,
    USER_PROMPT_FACTS_UPDATE,
    USER_PROMPT_PLAN,
    USER_PROMPT_PLAN_UPDATE,
)
from .tools import (
    DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
    Tool,
    get_tool_description_with_args,
)
from .types import AgentType


logger = getLogger(__name__)


def get_tool_descriptions(tools: Dict[str, Tool], tool_description_template: str) -> str:
    return "\n".join([get_tool_description_with_args(tool, tool_description_template) for tool in tools.values()])


def format_prompt_with_tools(tools: Dict[str, Tool], prompt_template: str, tool_description_template: str) -> str:
    tool_descriptions = get_tool_descriptions(tools, tool_description_template)
    prompt = prompt_template.replace("{{tool_descriptions}}", tool_descriptions)
    if "{{tool_names}}" in prompt:
        prompt = prompt.replace(
            "{{tool_names}}",
            ", ".join([f"'{tool.name}'" for tool in tools.values()]),
        )
    return prompt


def show_agents_descriptions(managed_agents: Dict):
    managed_agents_descriptions = """
You can also give requests to team members.
Calling a team member works the same as for calling a tool: the only argument you can give in the call is 'request', a long string explaining your request.
Given that this team member is a real human, you should be very verbose in your request.
Here is a list of the team members that you can call:"""
    for managed_agent in managed_agents.values():
        managed_agent: ManagedAgent = managed_agent
        managed_agents_descriptions += f"\n- {managed_agent.agent.name}: {managed_agent.description}"
    return managed_agents_descriptions


def format_prompt_with_managed_agents_descriptions(
    prompt_template,
    managed_agents: Dict[str, Any],
    agent_descriptions_placeholder: Optional[str] = None,
) -> str:
    if agent_descriptions_placeholder is None:
        agent_descriptions_placeholder = "{{managed_agents_descriptions}}"
    if agent_descriptions_placeholder not in prompt_template:
        raise ValueError(
            f"Provided prompt template does not contain the managed agents descriptions placeholder '{agent_descriptions_placeholder}'"
        )
    if len(managed_agents.keys()) > 0:
        agents_descriptions = show_agents_descriptions(managed_agents)
        return prompt_template.replace(agent_descriptions_placeholder, agents_descriptions)
    else:
        return prompt_template.replace(agent_descriptions_placeholder, "")


YELLOW_HEX = "#d4b702"


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        name (`str`): Name of the agent.
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        system_prompt (`str`, *optional*): System prompt that will be used to generate the agent's actions.
        tool_description_template (`str`, *optional*): Template used to describe the tools in the system prompt.
        max_steps (`int`, default `6`): Maximum number of steps the agent can take to solve the task.
        tool_parser (`Callable`, *optional*): Function used to parse the tool calls from the LLM output.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`int`, default `1`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
    """

    def __init__(
        self,
        name: str,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = None,
        max_steps: int = 6,
        tool_parser: Optional[Callable] = None,
        add_base_tools: bool = False,
        verbosity_level: int = 1,
        grammar: Optional[Dict[str, str]] = None,
        managed_agents: Optional[List] = None,
        step_callbacks: Optional[List[Callable]] = None,
        planning_interval: Optional[int] = None,
    ):
        self.name = name
        if system_prompt is None:
            system_prompt = PYTHON_CODE_SYSTEM_PROMPT
        if tool_parser is None:
            tool_parser = parse_json_tool_call
        self.agent_class = self.__class__.__name__
        self.model = model
        self.system_prompt_template = system_prompt
        self.tool_description_template = (
            tool_description_template if tool_description_template else DEFAULT_TOOL_DESCRIPTION_TEMPLATE
        )
        self.max_steps = max(100, max_steps)
        self.tool_parser = tool_parser
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state = {}

        managed_agents: List[ManagedAgent] = managed_agents if managed_agents is not None else []
        self.managed_agents: Dict[str, ManagedAgent] = {}
        if managed_agents is not None:
            self.managed_agents = {managed_agent.agent.name: managed_agent for managed_agent in managed_agents}

        for tool in tools:
            assert isinstance(tool, Tool), f"This element is not of class Tool: {str(tool)}"
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            for tool_name, tool_class in TOOL_MAPPING.items():
                if tool_name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent":
                    self.tools[tool_name] = tool_class()
                    
        self.tools["final_answer"] = FinalAnswerTool()

        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.task = None
        self.memory = AgentMemory(name, system_prompt)
        self.logger = AgentLogger(name, level=verbosity_level)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
            
    @property
    def logs(self):
        return [self.memory.system_prompt] + self.memory.steps

    def initialize_system_prompt(self):
        system_prompt = format_prompt_with_tools(
            self.tools,
            self.system_prompt_template,
            self.tool_description_template,
        )
        system_prompt = format_prompt_with_managed_agents_descriptions(system_prompt, self.managed_agents)
        return system_prompt

    def write_memory_to_messages(
        self,
        summary_mode: Optional[bool] = False,
    ) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def extract_action(self, model_output: str, split_token: str) -> Tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: Optional[list[str]]) -> str:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        messages = [{"role": MessageRole.SYSTEM, "content": []}]
        if images:
            messages[0]["content"] = [
                {
                    "type": "text",
                    "text": "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
                }
            ]
            messages[0]["content"].append({"type": "image"})
            messages += self.write_memory_to_messages()[1:]
            messages += [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Based on the above, please provide an answer to the following user request:\n{task}",
                        }
                    ],
                }
            ]
        else:
            messages[0]["content"] = [
                {
                    "type": "text",
                    "text": "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
                }
            ]
            messages += self.write_memory_to_messages()[1:]
            messages += [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Based on the above, please provide an answer to the following user request:\n{task}",
                        }
                    ],
                }
            ]
        try:
            chat_message: ChatMessage = self.model(messages)
            return chat_message.content
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"

    def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(arguments)
                else:
                    observation = available_tools[tool_name].__call__(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(**arguments)
                else:
                    observation = available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg, self.logger)
            return observation
        except Exception as e:
            if tool_name in self.tools:
                tool_description = get_tool_description_with_args(available_tools[tool_name])
                error_msg = (
                    f"Error in tool call execution: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following:\n{tool_description}"
                )
                raise AgentExecutionError(error_msg, self.logger)
            elif tool_name in self.managed_agents:
                error_msg = (
                    f"Error in calling team member: {e}\nYou should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{available_tools[tool_name]}"
                )
                raise AgentExecutionError(error_msg, self.logger)

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        pass

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        single_step: bool = False,
        images: Optional[List[str]] = None,
        additional_args: Optional[Dict] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            single_step (`bool`): Whether to run the agent in one-shot fashion.
            images (`list[str]`, *optional*): Paths to image(s).
            additional_args (`dict`): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """

        self.task = task
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
        self.logger.log(
            Panel(
                f"\n[bold]{self.task.strip()}\n",
                title=f"[bold]New run ({self.name})",
                subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=LogLevel.INFO,
        )

        self.memory.steps.append(TaskStep(task=self.task, task_images=images))
        if single_step:
            step_start_time = time.time()
            memory_step = ActionStep(start_time=step_start_time, observations_images=images)
            memory_step.end_time = time.time()
            memory_step.duration = memory_step.end_time - step_start_time

            # Run the agent's step
            result = self.step(memory_step)
            return result

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run(task=self.task, images=images)
        # Outputs are returned only at the end as a string. We only look at the last step
        return deque(self._run(task=self.task, images=images), maxlen=1)[0]

    def _run(self, task: str, images: List[str] | None = None) -> Generator[ActionStep | AgentType, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        final_answer = None
        self.step_number = 0
        while final_answer is None and self.step_number < self.max_steps:
            step_start_time = time.time()
            memory_step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            try:
                if self.planning_interval is not None and self.step_number % self.planning_interval == 0:
                    self.planning_step(
                        task,
                        is_first_step=(self.step_number == 0),
                        step=self.step_number,
                    )
                self.logger.log(
                    Rule(
                        f"[bold]Step {self.step_number}",
                        characters="━",
                        style=YELLOW_HEX,
                    ),
                    level=LogLevel.INFO,
                )

                # Run one step!
                final_answer = self.step(memory_step)
            except AgentError as e:
                memory_step.error = e
            finally:
                memory_step.end_time = time.time()
                memory_step.duration = memory_step.end_time - step_start_time
                self.memory.steps.append(memory_step)
                for callback in self.step_callbacks:
                    # For compatibility with old callbacks that don't take the agent as an argument
                    if len(inspect.signature(callback).parameters) == 1:
                        callback(memory_step)
                    else:
                        callback(memory_step, agent=self)
                self.step_number += 1
                yield memory_step

        if final_answer is None and self.step_number == self.max_steps:
            error_message = "Reached max steps."
            final_answer = self.provide_final_answer(task, images)
            final_memory_step = ActionStep(
                step_number=self.step_number, error=AgentMaxStepsError(error_message, self.logger)
            )
            final_memory_step = ActionStep(error=AgentMaxStepsError(error_message, self.logger))
            final_memory_step.action_output = final_answer
            final_memory_step.end_time = time.time()
            final_memory_step.duration = memory_step.end_time - step_start_time
            self.memory.steps.append(final_memory_step)
            for callback in self.step_callbacks:
                # For compatibility with old callbacks that don't take the agent as an argument
                if len(inspect.signature(callback).parameters) == 1:
                    callback(final_memory_step)
                else:
                    callback(final_memory_step, agent=self)
            yield final_memory_step

        yield handle_agent_output_types(final_answer)

    def planning_step(self, task, is_first_step: bool, step: int) -> None:
        """
        Used periodically by the agent to plan the next steps to reach the objective.

        Args:
            task (`str`): Task to perform.
            is_first_step (`bool`): If this step is not the first one, the plan should be an update over a previous plan.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        if is_first_step:
            message_prompt_facts = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_FACTS,
            }
            message_prompt_task = {
                "role": MessageRole.USER,
                "content": f"""Here is the task:
```
{task}
```
Now begin!""",
            }

            chat_message_facts: ChatMessage = self.model([message_prompt_facts, message_prompt_task])
            answer_facts = chat_message_facts.content

            message_system_prompt_plan = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_PLAN,
            }
            message_user_prompt_plan = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_PLAN.format(
                    task=task,
                    tool_descriptions=get_tool_descriptions(self.tools, self.tool_description_template),
                    managed_agents_descriptions=(show_agents_descriptions(self.managed_agents)),
                    answer_facts=answer_facts,
                ),
            }
            chat_message_plan: ChatMessage = self.model(
                [message_system_prompt_plan, message_user_prompt_plan],
                stop_sequences=["<end_plan>"],
            )
            answer_plan = chat_message_plan.content

            final_plan_redaction = f"""Here is the plan of action that I will follow to solve the task:
```
{answer_plan}
```"""
            final_facts_redaction = f"""Here are the facts that I know so far:
```
{answer_facts}
```""".strip()
            self.memory.steps.append(
                PlanningStep(
                    plan=final_plan_redaction,
                    facts=final_facts_redaction,
                    model_output_message_plan=chat_message_plan,
                    model_output_message_facts=chat_message_facts,
                )
            )
            self.logger.log(
                Rule("[bold]Initial plan", style="orange"),
                Text(final_plan_redaction),
                level=LogLevel.INFO,
            )
        else:  # update plan
            memory_messages = self.write_memory_to_messages(
                summary_mode=False
            )  # This will not log the plan but will log facts

            # Redact updated facts
            facts_update_system_prompt = {
                "role": MessageRole.SYSTEM,
                "content": [{"type": "text", "text": SYSTEM_PROMPT_FACTS_UPDATE}],
            }
            facts_update_message = {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": USER_PROMPT_FACTS_UPDATE}],
            }
            chat_message_facts: ChatMessage = self.model(
                [facts_update_system_prompt] + memory_messages + [facts_update_message]
            )
            facts_update = chat_message_facts.content

            # Redact updated plan
            plan_update_message = {
                "role": MessageRole.SYSTEM,
                "content": [{"type": "text", "text": SYSTEM_PROMPT_PLAN_UPDATE.format(task=task)}],
            }
            plan_update_message_user = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": USER_PROMPT_PLAN_UPDATE.format(
                            task=task,
                            tool_descriptions=get_tool_descriptions(self.tools, self.tool_description_template),
                            managed_agents_descriptions=(show_agents_descriptions(self.managed_agents)),
                            facts_update=facts_update,
                            remaining_steps=(self.max_steps - step),
                        ),
                    }
                ],
            }
            chat_message_plan: ChatMessage = self.model(
                [plan_update_message] + memory_messages + [plan_update_message_user],
                stop_sequences=["<end_plan>"],
            )

            # Log final facts and plan
            final_plan_redaction = PLAN_UPDATE_FINAL_PLAN_REDACTION.format(
                task=task, plan_update=chat_message_plan.content
            )
            final_facts_redaction = f"""Here is the updated list of the facts that I know:
```
{facts_update}
```"""
            self.memory.steps.append(
                PlanningStep(
                    plan=final_plan_redaction,
                    facts=final_facts_redaction,
                    model_output_message_plan=chat_message_plan,
                    model_output_message_facts=chat_message_facts,
                )
            )
            self.logger.log(
                Rule("[bold]Updated plan", style="orange"),
                Text(final_plan_redaction),
                level=LogLevel.INFO,
            )


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        system_prompt (`str`, *optional*): System prompt that will be used to generate the agent's actions.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        system_prompt: Optional[str] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = TOOL_CALLING_SYSTEM_PROMPT
        super().__init__(
            tools=tools,
            model=model,
            system_prompt=system_prompt,
            planning_interval=planning_interval,
            **kwargs,
        )

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages

        # Add new step in logs
        log_entry.model_input_messages = memory_messages.copy()

        try:
            model_message: ChatMessage = self.model(
                memory_messages,
                tools_to_call_from=list(self.tools.values()),
                stop_sequences=["Observation:"],
            )
            log_entry.model_output_message = model_message
            if model_message.tool_calls is None or len(model_message.tool_calls) == 0:
                raise Exception("Model did not call any tools. Call `final_answer` tool to return a final answer.")
            tool_call = model_message.tool_calls[0]
            tool_name, tool_call_id = tool_call.function.name, tool_call.id
            tool_arguments = tool_call.function.arguments

        except Exception as e:
            raise AgentGenerationError(f"Error in generating tool call with model:\n{e}", self.logger) from e

        log_entry.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]

        # Execute
        self.logger.log(
            Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
            level=LogLevel.INFO,
        )
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            if (
                isinstance(answer, str) and answer in self.state.keys()
            ):  # if the answer is a state variable, return the value
                final_answer = self.state[answer]
                self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                final_answer = answer
                self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )

            log_entry.action_output = final_answer
            return final_answer
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = self.execute_tool_call(tool_name, tool_arguments)
            observation_type = type(observation)
            if observation_type in [AgentImage]:
                if observation_type == AgentImage:
                    observation_name = "image.png"
                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."
            else:
                updated_information = str(observation).strip()
            self.logger.log(
                f"Observations: {updated_information.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            log_entry.observations = updated_information
            return None


class PythonAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
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
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        system_prompt: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        max_print_outputs_length: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = PYTHON_CODE_SYSTEM_PROMPT

        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        if "{{authorized_imports}}" not in system_prompt:
            raise ValueError("Tag '{{authorized_imports}}' should be provided in the prompt.")
        super().__init__(
            tools=tools,
            model=model,
            system_prompt=system_prompt,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                0,
            )

        all_tools = {**self.tools, **self.managed_agents}
        self.python_executor = LocalPythonInterpreter(
            self.additional_authorized_imports,
            all_tools,
            max_print_outputs_length=max_print_outputs_length,
        )

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
                    f"[italic]Output message of the LLM ({self.name}):",
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
        return output if is_final_answer else None



class SupervisorAgent:
    """
    SupervisorAgent class The LLM receives the final task report given by a Manager then it will return a review of the work done by the Manager.

    Args:
        company_name (`str`): The name of the company.
        sop (`str`): The standard operating procedure for the manager agent.
        agent (`object`): The agent to be managed.
        description (`str`): A description of the managed agent.
        additional_prompting (`str`, *optional*): Additional prompting for the managed agent, like 'add more detail in your answer'.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary after the agent completes its task. Defaults to False.
        use_redis (`bool`, *optional*): Whether to use Redis for communication. Defaults to True.
    """

    def __init__(
        self,
        company_name: str,
        sop: str,
        agent: MultiStepAgent,
        additional_prompting: Optional[str] = None,
        provide_run_summary: bool = False,
        use_redis: bool = True
    ):
        self.agent = agent
        self.sop = sop
        self.company_name = company_name
        self.additional_prompting = additional_prompting
        self.provide_run_summary = provide_run_summary
        self.supervisor_agent_prompt = SUPERVISOR_AGENT_PROMPT
        self.use_redis = use_redis
        self.task = None
        if use_redis:
            self.redis_client = Redis.from_url(os.environ["REDIS_URL"])

    def set_task(self, task: str):
        """Sets the task for the managed agent."""
        self.task = task
        
    def write_full_task(self, final_answer: str):
        """Adds additional prompting for the managed agent, like 'add more detail in your answer'."""
        full_task = self.supervisor_agent_prompt.format(name=self.agent.name, task=self.task, sop=self.sop, final_answer=final_answer)
        if self.additional_prompting:
            full_task = full_task.replace("\n{additional_prompting}", self.additional_prompting).strip()
        else:
            full_task = full_task.replace("\n{additional_prompting}", "").strip()
        return full_task

    def request(self, request, **kwargs):
        """Request the managed agent to perform a task."""
        return self.__call__(request, **kwargs)
    
    def __call__(self, request, **kwargs):
        if not isinstance(request, str):
            raise ValueError("Request must be a string.")
        if not self.task:
            raise ValueError("You need to set a task for the supervisor agent.")
        full_task = self.write_full_task(request)
        output = self.agent.run(full_task, **kwargs)
        if self.provide_run_summary:
            answer = f"Here is the final review from your supervisor '{self.agent.name}':\n"
            answer += str(output)
            answer += f"\n\nFor more detail, find below a summary of supervisor's review:\nSUMMARY OF REVIEW FROM SUPERVISOR '{self.agent.name}':\n"
            for message in self.agent.write_memory_to_messages(summary_mode=True):
                content = message["content"]
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += f"\nEND OF SUMMARY OF WORK FROM AGENT '{self.agent.name}'."
            output = {"answer": answer}
            
        if self.use_redis:
            output["agent"] = self.agent.name
            output_str = json.dumps(output)
            self.redis_client.publish(self.company_name, output_str)
            
        return output
    
def get_supervisor_agent_for_manager(model, company_name: str, sop: str):
    return SupervisorAgent(
        company_name=company_name,
        sop=sop,
        agent=PythonAgent(
            name="supervisoragent",
            tools=[],
            model=model,
            additional_authorized_imports=["*"],
            max_steps=3,
            verbosity_level=2,
        ),
    )
    
class ManagerAgent(MultiStepAgent):
    """
    In this agent, the LLM will always call a managed agent to perform the task. If no agent is available 
    then it will call the TextAgent to get a text response. It has no tools.

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
        sop: str,
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
        self.sop = sop
        managed_agents: List[ManagedAgent] = kwargs.pop("managed_agents", None)
        all_managed_agent: bool = all([isinstance(agent, ManagedAgent) for agent in managed_agents])
        if managed_agents is None:
            raise ValueError("You need to provide managed agents to the ManagerAgent.")
        if not all_managed_agent:
            raise ValueError("All agents in managed_agents should be of type ManagedAgent.")
        self.supervisor_agent = get_supervisor_agent_for_manager(model, company_name, sop)
        super().__init__(
            tools=self.tools,
            model=model,
            system_prompt=system_prompt,
            grammar=grammar,
            planning_interval=planning_interval,
            managed_agents=managed_agents,
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
        self.supervisor_agent.set_task(self.task)
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
            # TODO use review
            review = self.supervisor_agent(output)
            output_str = json.dumps({"answer": output, "agent": self.name})
            self.redis_client.publish(self.company_name, output_str)
        return output if is_final_answer else None


class ManagedAgent:
    """
    ManagedAgent class that manages an agent and provides additional prompting and run summaries.

    Args:
        company_name (`str`): The name of the company.
        agent (`object`): The agent to be managed.
        description (`str`): A description of the managed agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary after the agent completes its task. Defaults to False.
        managed_agent_prompt (`Optional[str]`, *optional*): Custom prompt for the managed agent. Defaults to None.

    """

    def __init__(
        self,
        company_name: str,
        agent: MultiStepAgent,
        description: str,
        additional_prompting: Optional[str] = None,
        provide_run_summary: bool = False,
        managed_agent_prompt: Optional[str] = None,
        use_redis: bool = True
    ):
        self.agent = agent
        self.description = description
        self.additional_prompting = additional_prompting
        self.provide_run_summary = provide_run_summary
        self.managed_agent_prompt = managed_agent_prompt
        if managed_agent_prompt is None:
            self.managed_agent_prompt = MANAGED_AGENT_PROMPT
        self.company_name = company_name
        self.use_redis = use_redis
        if use_redis:
            self.redis_client = Redis.from_url(os.environ["REDIS_URL"])

    def write_full_task(self, task):
        """Adds additional prompting for the managed agent, like 'add more detail in your answer'."""
        full_task = self.managed_agent_prompt.format(name=self.agent.name, task=task)
        if self.additional_prompting:
            full_task = full_task.replace("\n{additional_prompting}", self.additional_prompting).strip()
        else:
            full_task = full_task.replace("\n{additional_prompting}", "").strip()
        return full_task

    def request(self, request, **kwargs):
        """Request the managed agent to perform a task."""
        return self.__call__(request, **kwargs)
    
    def __call__(self, request, **kwargs):
        if not isinstance(request, str):
            raise ValueError("Request must be a string.")
        full_task = self.write_full_task(request)
        output = self.agent.run(full_task, **kwargs)
        if self.provide_run_summary:
            answer = f"Here is the final answer from your managed agent '{self.agent.name}':\n"
            answer += str(output)
            answer += f"\n\nFor more detail, find below a summary of this agent's work:\nSUMMARY OF WORK FROM AGENT '{self.agent.name}':\n"
            for message in self.agent.write_memory_to_messages(summary_mode=True):
                content = message["content"]
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += f"\nEND OF SUMMARY OF WORK FROM AGENT '{self.agent.name}'."
            output = {"answer": answer}
        if self.use_redis:
            output["agent"] = self.agent.name
            output_str = json.dumps(output)
            self.redis_client.publish(self.company_name, output_str)
        return output


__all__ = ["ManagedAgent", "MultiStepAgent", "PythonAgent", "ToolCallingAgent", "ManagerAgent", "AgentMemory"]
