import requests
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class TfServingExecutor:
    """
    A local TensorFlow Serving executor that performs HTTP requests for model inference.

    This environment accepts input data for prediction and executes a REST API call to
    TensorFlow Serving to obtain inference results.
    """
    def __init__(self, base_url: str, model_name: str, signature_name: str = "serving_default"):
        """
        Initialize the executor with the base URL, model name, and an optional signature name.

        Args:
            base_url (str): The base URL for the TensorFlow Serving REST API.
            model_name (str): The name of the model to query.
            signature_name (str, optional): The signature name to use for inference. Defaults to "serving_default".
        """
        self.base_url = base_url
        self.model_name = model_name
        self.signature_name = signature_name

    def __call__(self, input_data: Any, additional_variables: Dict = None) -> Dict:
        """
        Executes a prediction request on the TensorFlow Serving model.

        Args:
            input_data (Any): The input data for inference, typically a list of instances.
            additional_variables (Dict, optional): Additional parameters to include in the request payload.

        Returns:
            Dict: The JSON response from TensorFlow Serving containing the inference results.
        """
        url = f"{self.base_url}/v1/models/{self.model_name}:predict"
        payload = {
            "instances": input_data,
            "signature_name": self.signature_name
        }
        if additional_variables:
            payload.update(additional_variables)

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an error for bad HTTP status codes.
            return response.json()
        except Exception as e:
            logger.error(f"Error during TensorFlow Serving request: {e}")
            return {}
