import json
import os
from agentcompany.llms.base import AugmentedLLM
from typing import Dict, List, Optional, Type, Union, Tuple, Any
from agentcompany.llms.utils import ChatMessage
from agentcompany.llms.base import ReturnType, Argument
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ArrayOutput(BaseModel):
    array: list[str]
    
class OpenAIServerLLM(AugmentedLLM):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] | None = None,
        project: Optional[str] | None = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        try:
            import openai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'openai'"
            ) from None

        super().__init__(**kwargs)
        self.model_id = model_id
        if self.model_id == "deepseek-chat":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            api_base = "https://api.deepseek.com"
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            organization=organization,
            project=project,
        )
        
        # logfire.instrument_openai(self.client)  
        self.custom_role_conversions = custom_role_conversions

    def __call__(
        self,
        messages: List[Dict[str, str]],
        return_type: ReturnType = "string",
        **kwargs,
    ) -> ChatMessage:
        # TODO add return type
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        
        if return_type == "list":
            completion_kwargs["response_format"] = ArrayOutput
            value = self.client.beta.chat.completions.parse(**completion_kwargs).choices[0].message.parsed
            print(f"list value: {value}")
            value = value.model_dump(include={"array"})
            print(f"list value: {value}")
            message = {
                "role": "assistant",
                "content": value["array"],
            }
            print(f"list message: {message}")
            message = ChatMessage.from_dict(
                message
            )
        elif return_type == "string":
            response = self.client.chat.completions.create(**completion_kwargs)
            message = ChatMessage.from_dict(
                response.choices[0].message.model_dump(include={"role", "content"})
            )
            message.raw = response
        else:
            raise ValueError(
                f"Invalid return_type '{return_type}'. Supported types are 'string' and 'list'."
            )
        # self.last_input_token_count = response.usage.prompt_tokens
        # self.last_output_token_count = response.usage.completion_tokens        
        return message

    def generate_system_prompt(self, model_cls: type[BaseModel]) -> str:
        """
        Generates a system prompt for an LLM to output a JSON string
        that strictly conforms to the given Pydantic model schema.
        
        Parameters:
            model_cls (Type[BaseModel]): A Pydantic BaseModel subclass.
            
        Returns:
            str: The generated system prompt.
        """
        # Get the JSON schema from the Pydantic model
        schema_json = model_cls.model_json_schema()
        
        # Create the prompt using the dynamically generated schema.
        prompt = f"""
        You are a highly accurate and detail-oriented assistant. When you respond, you must output a single valid JSON object that strictly adheres to the following JSON schema:

        {schema_json}

        Important:
        1. Output only the JSON object without any additional text, explanation, markdown formatting, or code fences.
        2. Do not include any keys or values that are not defined in the schema.
        3. The JSON must be parsable by standard JSON parsers.
        4. No extra commentary or wrapping text is allowed; your entire response must be exactly the JSON string.

        Now, please generate the JSON output that conforms exactly to the schema above.
        """.strip()
        
        return prompt
    
    def function_call(self, prompt: str, name: str, description: str, argument_list: List[Argument]) -> Union[Tuple[str, Dict[str, str]], None]:
        
        output_schema = {
            "type": "object",
            "properties": {
                arg["name"]: {
                    "type": "string",
                    "description": arg["description"]
                } for arg in argument_list
            }
        }
        
        tool = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": output_schema,
                "required": [arg["name"] for arg in argument_list],
                "additionalProperties": False
            }
        }
        logger.info(f"Tool: {tool}")
        response = self.client.responses.create(
            model=self.model_id,
            input=[{"role": "user", "content": "Can you send an email to ilan@example.com and katia@example.com saying hi?"}],
            tools=[tool]
        )
        return response.output

    def structured_output(
        self,
        prompt: str,
        output_schema: Type[BaseModel],
        **kwargs,
    ) -> Union[BaseModel, None]:
        """
        Sends the provided prompt to the model and expects a JSON-formatted response.
        This JSON is then parsed into an instance of the provided Pydantic model class.

        Parameters:
            prompt (str): The prompt to send to the model.
            output_schema (Type[BaseModel]): A Pydantic model class that defines the expected structure.
            **kwargs: Additional keyword arguments passed to the underlying model call.

        Returns:
            An instance of output_schema populated with the model's output.
        """
        system_prompt = self.generate_system_prompt(output_schema)
        # Create a simple message list with the prompt.
        complete_prompt = f"""
        {system_prompt}
        {prompt}
        """
        messages = [{"role": "user", "content": complete_prompt}]
        response = self.__call__(messages, **kwargs)
        attempt = 0
        while attempt < 3:
            try:
                parsed_content = json.loads(response.content)
                return output_schema.model_validate(parsed_content)
            except Exception as e:
                logger.error(f"Failed to parse response content as JSON: {e}")
                attempt += 1
        return None
