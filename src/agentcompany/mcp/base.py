from typing import Dict, Union, Optional
from functools import wraps
from ._function_type_hints_utils import (
    _convert_type_hints_to_json_schema,
)
import abc
import inspect

AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
]

def validate_after_init(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments()

    cls.__init__ = new_init
    return cls


class ModelContextProtocolImpl(abc.ABC):
    """
    A base class for the functions used by the agent. Subclass this and implement the `forward` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your function does, the inputs it expects and the output(s) it
      will return. For instance 'This is a function that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **name** (`str`) -- A performative name that will be used for your function in the prompt to the agent. For instance
      `"text-classifier"` or `"image_generator"`.
    - **inputs** (`Dict[str, Dict[str, Union[str, type]]]`) -- The dict of modalities expected for the inputs.
      It has one `type`key and a `description`key.
      This is used by `launch_gradio_demo` or to make a nice space from your function, and also can be used in the generated
      description for your function.
    - **output_type** (`type`) -- The type of the function output. This is used by `launch_gradio_demo`
      or to make a nice space from your function, and also can be used in the generated description for your function.

    You can also override the method [`~ModelContextProtocolImpl.setup`] if your function has an expensive operation to perform before being
    usable (such as loading a model). [`~ModelContextProtocolImpl.setup`] will be called the first time you use your function, but not at
    instantiation.
    """

    name: str
    description: str
    inputs: Dict[str, Dict[str, Union[str, type, bool]]]
    output_type: str

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }

        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )
        for input_name, input_content in self.inputs.items():
            assert isinstance(input_content, dict), f"Input '{input_name}' should be a dictionary."
            assert "type" in input_content and "description" in input_content, (
                f"Input '{input_name}' should have keys 'type' and 'description', has only {list(input_content.keys())}."
            )
            if input_content["type"] not in AUTHORIZED_TYPES:
                raise Exception(
                    f"Input '{input_name}': type '{input_content['type']}' is not an authorized value, should be one of {AUTHORIZED_TYPES}."
                )

        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES

        # Validate forward function signature, except for Tools that use a "generic" signature (PipelineTool, SpaceToolWrapper, LangChainToolWrapper)
        if not (
            hasattr(self, "skip_forward_signature_validation")
            and getattr(self, "skip_forward_signature_validation") is True
        ):
            signature = inspect.signature(self.forward)

            if not set(signature.parameters.keys()) == set(self.inputs.keys()):
                raise Exception(
                    f"In tool '{self.name}', 'forward' method should take 'self' as its first argument, then its next arguments should match the keys of tool attribute 'inputs'."
                )

            json_schema = _convert_type_hints_to_json_schema(self.forward, error_on_missing_type_hints=False)[
                "properties"
            ]  # This function will not raise an error on missing docstrings, contrary to get_json_schema
            for key, value in self.inputs.items():
                assert key in json_schema, (
                    f"Input '{key}' should be present in function signature, found only {json_schema.keys()}"
                )
                if "nullable" in value:
                    assert "nullable" in json_schema[key], (
                        f"Nullable argument '{key}' in inputs should have key 'nullable' set to True in function signature."
                    )
                if key in json_schema and "nullable" in json_schema[key]:
                    assert "nullable" in value, (
                        f"Nullable argument '{key}' in function signature should have key 'nullable' set to True in inputs."
                    )

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        return NotImplementedError("Write this method in your subclass of `ModelContextProtocolImpl`.")

    def __call__(self, *args, **kwargs):
        if not self.is_initialized:
            self.setup()
        # Handle the arguments might be passed as a single dictionary
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]

            # If the dictionary keys match our input parameters, convert it to kwargs
            if all(key in self.inputs for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        outputs = self.forward(*args, **kwargs)
        return outputs

    def setup(self):
        """
        Overwrite this method here for any operation that is expensive and needs to be executed before you start using
        your function. Such as loading a big model.
        """
        self.is_initialized = True
    