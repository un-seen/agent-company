import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type
from agentcompany.mcp.base import ModelContextProtocolImpl

from ..mcp.utils import encode_image_base64, make_image_url

# Import BaseModel from Pydantic for structured output parsing.
from pydantic import BaseModel


logger = logging.getLogger(__name__)


DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_code>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_code>",
}


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessageFunctionCallDefinition:
    arguments: Any
    name: str
    description: Optional[str] = None


@dataclass
class ChatMessageFunctionCall:
    function: ChatMessageFunctionCallDefinition
    id: str
    type: str

@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    function_calls: Optional[List[ChatMessageFunctionCall]] = None
    raw: Optional[Any] = None  # Stores the raw output from the API

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("function_calls"):
            function_calls = [
                ChatMessageFunctionCall(
                    function=ChatMessageFunctionCallDefinition(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["function_calls"]
            ]
            data["function_calls"] = function_calls
        return cls(**data)

    def dict(self):
        return json.dumps(get_dict_from_nested_dataclasses(self))


def parse_json_if_needed(arguments: Union[str, dict]) -> Union[str, dict]:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


def parse_function_args_if_needed(message: ChatMessage) -> ChatMessage:
    for function_call in message.function_calls:
        function_call.function.arguments = parse_json_if_needed(function_call.function.arguments)
    return message


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION_CALL = "function-call"
    FUNCTION_RESPONSE = "function-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


function_role_conversions = {
    MessageRole.FUNCTION_CALL: MessageRole.ASSISTANT,
    MessageRole.FUNCTION_RESPONSE: MessageRole.USER,
}


def get_function_json_schema(function: ModelContextProtocolImpl) -> Dict:
    properties = deepcopy(function.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": function.name,
            "description": function.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: List[Dict[str, str]],
    role_conversions: Dict[MessageRole, MessageRole] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> List[Dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

    Args:
        message_list (`list[dict[str, str]]`): List of chat messages.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]
        # encode images if needed
        if isinstance(message["content"], list):
            for i, element in enumerate(message["content"]):
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        message["content"][i] = {
                            "type": "image_url",
                            "image_url": {"url": make_image_url(encode_image_base64(element["image"]))},
                        }
                    else:
                        message["content"][i]["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message["role"] == output_message_list[-1]["role"]:
            assert isinstance(message["content"], list), "Error: wrong content:" + str(message["content"])
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += message["content"][0]["text"]
            else:
                output_message_list[-1]["content"] += message["content"]
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list
