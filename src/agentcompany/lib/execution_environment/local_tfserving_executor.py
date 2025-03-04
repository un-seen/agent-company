import requests
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from PIL import Image
import numpy as np
import io
        
DEFAULT_MAX_LEN_OUTPUT = 5000


class TfServingExecutorInput(TypedDict):
    endpoint: str
    file_url: str    


class TFServingExecutor:
    """
    A simple executor class for TensorFlow Serving that takes as input:
      - A dictionary of TFServing endpoints (model_name: endpoint_url).
      - An action dictionary with 'model_name' and 'file_url' to process.

    The executor fetches the file from the URL, preprocesses it (placeholder logic),
    and sends it to the specified TFServing endpoint via REST API.

    **Warning:** This is a simplified proof-of-concept. Real-world use requires
    proper file validation, preprocessing, and error handling.
    """
    
    def __init__(
        self,
        tfserving_endpoints: Optional[Dict[str, str]] = None,
        allowed_models: Optional[List[str]] = None,
        max_print_outputs_length: Optional[int] = None,
    ):
        # Dictionary of TFServing endpoints (model_name: endpoint_url).
        self.endpoints = tfserving_endpoints if tfserving_endpoints is not None else {}
        
        # Define a whitelist of allowed model names.
        if allowed_models is None:
            self.allowed_models = list(self.endpoints.keys()) if self.endpoints else []
        else:
            self.allowed_models = allowed_models

        self.state: Dict[str, Any] = {}  # Additional state variables (if needed)
        self.max_print_outputs_length = max_print_outputs_length or DEFAULT_MAX_LEN_OUTPUT
        self.print_outputs = ""  # Log buffer for all command outputs

    def _is_model_allowed(self, model_name: str) -> bool:
        """
        Check if the specified model name is in the whitelist.
        """
        return model_name in self.allowed_models

    def _fetch_file(self, file_url: str) -> bytes:
        """
        Fetch the file content from the given URL.
        """
        try:
            response = requests.get(file_url, timeout=10)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch file from {file_url}: {str(e)}")

    def _preprocess_file(self, file_name: str, file_content: bytes, model_name: str) -> List[np.ndarray]:
        """
        Placeholder for preprocessing the file content based on the model.
        Replace with actual logic (e.g., image resizing, text tokenization).
        """
        # Load image from bytes
        try:
            image = Image.open(io.BytesIO(file_content)).convert('RGB')
            # Resize to a common input size (e.g., 224x224 for many image models)
            image = image.resize((512, 512))
            # Convert to numpy array and normalize to [0, 1]
            image_array = np.array(image) / 255.0
            # Convert to list of lists (tensor shape: [224, 224, 3] for RGB)
            tensor = image_array.tolist()
            # Return as a list of one tensor (batch size 1)
            print(f"Preprocessed image with shape {image_array.shape}")
            return [tensor]
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def __call__(self, action: TfServingExecutorInput, additional_variables: Dict[str, Any] = None) -> Tuple[Any, str, bool]:
        """
        Executes the TFServing prediction request based on the provided action.

        Args:
            action (Dict[str, Any]): A dictionary with 'model_name' and 'file_url'.
            additional_variables (Dict[str, Any], optional): Additional environment variables
                (not used in this executor but kept fofr compatibility).

        Returns:
            Tuple[str, str, bool]:
              - The prediction output from TFServing (as a JSON string),
              - The accumulated log of outputs,
              - A boolean flag is_final_answer (always True for this executor).
        """
        # Extract model_name and file_url from the action dictionary
        try:
            model_name = action["model_name"]
            file_url = action["file_url"]
        except KeyError as e:
            raise ValueError(f"Action dictionary missing required key: {str(e)}")

        # Validate the model name
        if not self._is_model_allowed(model_name):
            raise ValueError(f"Model '{model_name}' is not allowed. Allowed models: {self.allowed_models}")

        # Get the corresponding endpoint
        endpoint = self.endpoints.get(model_name)
        if not endpoint:
            raise ValueError(f"No endpoint defined for model '{model_name}'")

        # Fetch and preprocess the file
        file_content = self._fetch_file(file_url)
        input_data = self._preprocess_file(file_url, file_content, model_name)

        # Prepare the TFServing payload
        payload = {"instances": input_data}

        # Execute the TFServing request
        try:
            print(f"Sending request to {endpoint}")
            response = requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()
            output = response.json()  # Return raw JSON string
        except requests.exceptions.RequestException as e:
            raise ValueError(f"TFServing request failed for {endpoint}: {str(e)}")

        # Append to log buffer (truncate if necessary)
        self.print_outputs += str(output) + "\n"
        if len(self.print_outputs) > self.max_print_outputs_length:
            self.print_outputs = self.print_outputs[-self.max_print_outputs_length:]

        # For TFServing, the prediction is considered the final answer
        is_final_answer = True

        return output, self.print_outputs, is_final_answer
