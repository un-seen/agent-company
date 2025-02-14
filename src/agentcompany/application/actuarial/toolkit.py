from agentcompany.driver import PythonCodeAgent, OpenAIServerModel, tool, ManagedAgent
from agentcompany.driver.agents import ActionStep
from .protocol import FileDictionary
from .mongodb import mongodb_connect
import pandas as pd
from typing import List, Generator
from PIL import Image
import numpy as np
    
def index_file(model: OpenAIServerModel, file_path: str) -> None:
    # Check file is an image or a csv
    # if image, return image metadata using the model with call_prompt_with_image api
    # if csv, return data dictionary using the model with call_prompt_with_structured_output api
    # if neither, return error
    if file_path.endswith(".csv"):
        # Sample random 10 rows from the csv file
        data = pd.read_csv(file_path)
        data_sample = data.sample(10)
        prompt = f"""
        Give me the data dictionary for `{file_path}` where a name, definition and type system is available for every set of column values.
        
        Here is the sampled data from the file `{file_path}`:
        {data_sample}
        """
        
        output = model.structured_output(prompt, FileDictionary)
        if output is None:
            raise ValueError("Model did not return a structured output")
        from .mongodb import mongodb_connect
        system_client = mongodb_connect()
        system_state = system_client["agentos"]
        collection = system_state["datalake"]
        document_dict = output.model_dump()
        document_dict['_id'] = file_path
        collection.insert_one(document_dict)
    else:
        raise ValueError("File type not supported")


@tool
def get_file_dictionary(path: str) -> str:
    """
    It loads csv data into a pandas dataframe
    Args:
        path: The path to the data for the csv
    Returns:
        df: a pandas dataframe
    """
    system_client = mongodb_connect()
    system_state = system_client["agentos"]
    collection = system_state["datalake"]
    document = collection.find_one({"_id": path})
    file_dictionary = FileDictionary.model_validate(document)
    return file_dictionary.to_dataframe()

def get_datalake() -> Generator[str, None, None]:
    """
    It loads csv data into a pandas dataframe
    Args:
        path: The path to the data for the csv
    Returns:
        df: a pandas dataframe
    """
    system_client = mongodb_connect()
    system_state = system_client["agentos"]
    collection = system_state["datalake"]
    project = {"_id": 1}
    document = collection.find({}, project)
    for doc in document:
        yield doc["_id"]

def select_file(model: OpenAIServerModel, file_path: List[str]) -> None:
    # Check file is an image or a csv
    # if image, return image metadata using the model with call_prompt_with_image api
    # if csv, return data dictionary using the model with call_prompt_with_structured_output api
    # if neither, return error
    if file_path.endswith(".csv"):
        # Sample random 10 rows from the csv file
        data = pd.read_csv(file_path)
        data_sample = data.sample(10)
        prompt = f"""
        Give me the data dictionary for `{file_path}` where a name, definition and type system is available for every set of column values.
        
        Here is the sampled data from the file `{file_path}`:
        {data_sample}
        """
        
        output = model.structured_output(prompt, FileDictionary)
        if output is None:
            raise ValueError("Model did not return a structured output")
        from .mongodb import mongodb_connect
        system_client = mongodb_connect()
        system_state = system_client["agentos"]
        collection = system_state["datalake"]
        document_dict = output.model_dump()
        document_dict['_id'] = file_path
        collection.insert_one(document_dict)
    else:
        raise ValueError("File type not supported")
    
@tool
def get_csv_as_dataframe(path: str) -> str:
    """
    It loads csv data into a pandas dataframe
    Args:
        path: The path to the data for the csv
    Returns:
        df: a pandas dataframe
    """
    return pd.read_csv(path)

@tool 
def generate_image_statistics(image_path: str) -> dict:
    """
    Generate basic color statistics for an input image.

    Args:
        image_path: The string file path to the input image.

    Returns:
        dict: A dictionary containing statistics of the image, including mean,
              median, standard deviation, min, and max for each color channel.
    """
    # Open the image and convert it to RGB (if it's not already)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Compute statistics along the height and width dimensions, leaving the channel axis intact
    mean = np.mean(img_array, axis=(0, 1))
    median = np.median(img_array, axis=(0, 1))
    std_dev = np.std(img_array, axis=(0, 1))
    min_val = np.min(img_array, axis=(0, 1))
    max_val = np.max(img_array, axis=(0, 1))
    
    # Create a dictionary of the computed statistics
    stats = {
        'mean': {
            'red': mean[0],
            'green': mean[1],
            'blue': mean[2]
        },
        'median': {
            'red': median[0],
            'green': median[1],
            'blue': median[2]
        },
        'std_dev': {
            'red': std_dev[0],
            'green': std_dev[1],
            'blue': std_dev[2]
        },
        'min': {
            'red': min_val[0],
            'green': min_val[1],
            'blue': min_val[2]
        },
        'max': {
            'red': max_val[0],
            'green': max_val[1],
            'blue': max_val[2]
        }
    }
    
    return stats
