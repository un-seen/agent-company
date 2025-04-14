from agentcompany.llms.utils import (
    function_role_conversions,
    get_clean_message_list,
    get_function_json_schema,
)
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.llms.utils import ChatMessage
from typing import Dict, List, Optional, Callable, Literal

BaseLLM = Callable[[List[Dict[str, str]]], ChatMessage]
ReturnType = Literal["list", "string"]

class AugmentedLLM:
    def __init__(self, **kwargs):
        self.last_input_token_count = None
        self.last_output_token_count = None
        # Set default values for common parameters
        kwargs.setdefault("max_tokens", 4096)
        self.kwargs = kwargs

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for LLM invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or function_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def get_token_counts(self) -> Dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def __call__(
        self,
        messages: List[Dict[str, str]],
        return_type: ReturnType = "string",
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the LLM's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the LLM's output.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the LLM's response.
            functions_to_call_from (`List[ModelContextProtocolImpl]`, *optional*):
                A list of functions that the LLM can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying LLM.

        Returns:
            `ChatMessage`: A chat message object containing the LLM's response.
        """
        raise NotImplementedError("The __call__ method must be implemented in a child class.")
