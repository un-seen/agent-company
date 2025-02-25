from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict, Union
import os
from .models import ChatMessage, MessageRole
from .utils import AgentError, make_json_serializable
import os
from typing import TypeVar, Generic, Optional, Iterator
from redis import Redis

if TYPE_CHECKING:
    from .models import ChatMessage


logger = getLogger(__name__)


class Message(TypedDict):
    role: MessageRole
    content: str | list[dict]


def get_memory_index_name() -> str:
    avatar_id = os.getenv("AVATAR_ID")
    index_name = f"{avatar_id}-index"
    return index_name

@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


class MemoryStep:
    raw: Any  # This is a placeholder for the raw data that the agent logs

    def dict(self):
        return asdict(self)

    def to_messages(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    model_input_messages: List[Dict[str, str]] | None = None
    tool_calls: List[ToolCall] | None = None
    start_time: float | None = None
    end_time: float | None = None
    step_number: int | None = None
    error: AgentError | None = None
    duration: float | None = None
    model_output_message: ChatMessage = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: List[str] | None = None
    action_output: Any = None

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "step": self.step_number,
            "error": self.error.dict() if self.error else None,
            "duration": self.duration,
            "model_output_message": self.model_output_message,
            "model_output": self.model_output,
            "observations": self.observations,
            "action_output": make_json_serializable(self.action_output),
        }

    def to_messages(self, summary_mode: bool = False, show_model_input_messages: bool = False) -> List[Dict[str, Any]]:
        messages = []
        if self.model_input_messages is not None and show_model_input_messages:
            messages.append(Message(role=MessageRole.SYSTEM, content=self.model_input_messages))
        if self.model_output is not None and not summary_mode:
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[{"type": "text", "text": str([tc.dict() for tc in self.tool_calls])}],
                )
            )

        if self.error is not None:
            message_content = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            if self.tool_calls is None:
                tool_response_message = Message(
                    role=MessageRole.ASSISTANT, content=[{"type": "text", "text": message_content}]
                )
            else:
                tool_response_message = Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[{"type": "text", "text": f"Call id: {self.tool_calls[0].id}\n{message_content}"}],
                )

            messages.append(tool_response_message)
        else:
            if self.observations is not None and self.tool_calls is not None:
                messages.append(
                    Message(
                        role=MessageRole.TOOL_RESPONSE,
                        content=[
                            {
                                "type": "text",
                                "text": f"Call id: {self.tool_calls[0].id}\nObservation:\n{self.observations}",
                            }
                        ],
                    )
                )
        if self.observations_images:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": "Here are the observed images:"}]
                    + [
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )
        return messages


@dataclass
class PlanningStep(MemoryStep):
    model_output_message_facts: ChatMessage
    facts: str
    model_output_message_facts: ChatMessage
    plan: str

    def to_messages(self, summary_mode: bool, **kwargs) -> List[Dict[str, str]]:
        messages = []
        messages.append(Message(role=MessageRole.ASSISTANT, content=f"[FACTS LIST]:\n{self.facts.strip()}"))

        if not summary_mode:
            messages.append(Message(role=MessageRole.ASSISTANT, content=f"[PLAN]:\n{self.plan.strip()}"))
        return messages


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: List[str] | None = None

    def to_messages(self, summary_mode: bool, **kwargs) -> List[Dict[str, str]]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            for image in self.task_images:
                content.append({"type": "image", "image": image})

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool, **kwargs) -> List[Dict[str, str]]:
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt.strip()}])]


T = TypeVar('T')

class StepsList(Generic[T]):
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.items: List[T] = []
        import os
        from redis import Redis
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
        agent_index_name = get_memory_index_name()
        self.agent_index_name = agent_index_name
        self.items = []
        
    def append(self, item):
        self.add(item)
        
    def reset(self):
        self.items = []
    
    def save_step_in_memory(self, task_step: Union[TaskStep, ActionStep, PlanningStep]) -> None:
        content = None
        step_type = None
        if isinstance(task_step, TaskStep):
            content = task_step.task
            step_type = "task"
        elif isinstance(task_step, ActionStep):
            def dict_to_str(d: dict) -> str: 
                output_str = ""
                for k, v in d.items():
                    output_str += f"{k}: {v}\n"
                return output_str
            model_input = [dict_to_str(m) for m in task_step.model_input_messages]
            model_input = "\n".join(model_input)
            content = f"INPUT: {model_input} \n\n OUTPUT: {task_step.model_output}"
            step_type = "action"
        elif isinstance(task_step, PlanningStep):
            content = task_step.facts + "\n\n" + task_step.plan
            step_type = "planning"
            
        self.redis_client.append(self.agent_index_name, {
            "content": content,
            "step_type": step_type,
            "name": self.name,
            "input_type": "passage", 
        })
    
    def add(self, item: T) -> None:
        # self.embed(item)
        self.items.append(item)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[T, List[T]]:
        return self.items[index]
    
    def __setitem__(self, index: Union[int, slice], value: Union[T, List[T]]) -> None:
        self.items[index] = value
    
    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.items[index]
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self) -> Iterator[T]:
        return iter(self.items)
    
    def __contains__(self, item: T) -> bool:
        return item in self.items
    
    def extend(self, iterable: List[T]) -> None:
        self.items.extend(iterable)
    
    def insert(self, index: int, item: T) -> None:
        self.items.insert(index, item)
    
    def remove(self, item: T) -> None:
        self.items.remove(item)
    
    def pop(self, index: int = -1) -> T:
        return self.items.pop(index)
    
    def clear(self) -> None:
        self.items.clear()
    
    def index(self, item: T, start: int = 0, end: Optional[int] = None) -> int:
        if end is None:
            end = len(self.items)
        return self.items.index(item, start, end)
    
    def count(self, item: T) -> int:
        return self.items.count(item)
    
    def sort(self, *, key=None, reverse: bool = False) -> None:
        self.items.sort(key=key, reverse=reverse)
    
    def reverse(self) -> None:
        self.items.reverse()
    
    def copy(self) -> "StepsList[T]":
        new_list = StepsList[T]()
        new_list.items = self.items.copy()
        return new_list

    # Overriding addition and multiplication operators:
    
    def __add__(self, other: List[T]) -> List[T]:
        return self.add(other)
    
    def __iadd__(self, other: List[T]) -> "StepsList[T]":
        self.add(other)
        return self
    
    def __str__(self) -> str:
        return str(self.items)
        
    def __repr__(self) -> str:
        return f"StepsList({self.items})"
        

class AgentMemory:
    def __init__(self, name: str, company_name: str, system_prompt: str):
        self.name = name
        self.company_name = company_name
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: StepsList[Union[TaskStep, ActionStep, PlanningStep]] = StepsList(name)

    def reset(self):
        self.steps.reset()

    def get_succinct_steps(self) -> list[dict]:
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]


__all__ = ["AgentMemory"]
