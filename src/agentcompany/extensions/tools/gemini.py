from google import genai
import os
import re
from typing import Optional, List
import pandas as pd
from pydantic import BaseModel
from agentcompany.extensions.tools.brave import brave_web_search
from agentcompany.extensions.tools.jina import get_url_as_text

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

class Column(BaseModel):
    """
    A row in a table.
    """
    key: str
    value: str
    
    
class Row(BaseModel):
    """
    A row in a table.
    """
    key_list: List[str]
    list_of_value_list: List[List[str]]
    
    
def convert_list_row_to_dataframe(row: Row) -> pd.DataFrame:
    """
    Convert a list of Row objects to a pandas DataFrame.
    """
    data = []
    headers = row.key_list
    for column_value in row.list_of_value_list:
        row_dict = {}
        for idx, column_name in enumerate(headers):
            row_dict[column_name] = column_value[idx]
        data.append(row_dict)
    
    df = pd.DataFrame(data)
    
    return df


def generate_json_array_from_text(user_text: str, web_text: str) -> Row:
    """
    Generate a JSON array for the user text using the Gemini API.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=f'Generate a dataset that is relevant to user query: {user_text} \n\n from this text: \n\n {web_text}',
        config={
            'response_mime_type': 'application/json',
            'response_schema': Row,
        },
    )
    response: List[Row] = response.parsed
    return response

def get_web_text(prompt: str) -> str:
    """
    Generate a JSON array for the user text using the Gemini API.
    """
    url_list = brave_web_search(prompt)
    web_data: List[str] = []
    limit = 1
    for index, url in enumerate(url_list):
        if index >= limit:
            break
        web_text = get_url_as_text(url['url'])
        web_data.append(web_text)
    return "\n\n".join(web_data)

def get_identifier_value(context: str, identifier: str, data: str) -> Optional[str]:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    # Find the sentence which has the identifier 
    pattern = r"([^.]*?\b" + re.escape(identifier) + r"\b[^.]*\.)"
    sentences = re.findall(pattern, context)
    prefix = ""
    if sentences:
        sentence = sentences[0]
        # Find the prefix in the sentence
        index = sentence.find(identifier)
        if index != -1:
            prefix = sentence[:index-2].strip()
    prompt = f"""
    You have to complete this sentence with a plain text value and not formatting\n\n
    {prefix} _____ \n\n
    
    
    You can only use the following data to compute the value for the blank: \n\n
    {data} \n\n
    """
        
    print(prompt)
    response = client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=prompt
    )
    value = response.text
    pattern = r"VALUE:(.*?):END"
    match = re.search(pattern, value)
    if match:
        return match.group(1)
    return None    