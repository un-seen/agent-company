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

def get_web_text(context: str, identifier: str) -> str:
    """
    Generate a JSON array for the user text using the Gemini API.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=f'Write a question which has the answer as the identifier: {identifier} \n\n in this text: \n\n {context}.',
    )
    question = response.text
    url_list = brave_web_search(question)
    web_data: List[str] = []
    for url in url_list:
        web_text = get_url_as_text(url['url'])
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=f'Decide whether this ground truth data has sufficient, necessary or no evidence to answer the queestion: \n\n {web_text}.',
            config={
                'response_mime_type': 'text/x.enum',
                'response_schema': {
                    "type": "STRING",
                    "enum": [ "sufficient", "necessary", "insufficient", "none" ],
                }
            },
        )
        if response.text == "sufficient":
            web_data.append(f"\n\n{url['title']} ({url['url']}):\n{web_text}")
            break
        elif response.text == "necessary":
            web_data.append(f"\n\n{url['title']} ({url['url']}):\n{web_text}")
    return web_data

def get_identifier_value(context: str, identifier: str, data: str) -> Optional[str]:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=f"""
        Generate the value for the {identifier} in the context: {context} \n\n
        
        You can only use the following data to answer the question: \n\n
        {data} \n\n
        
        Please prefix the value with "VALUE: " and suffix with ":END" \n\n
        """
    )
    value = response.text
    pattern = r"VALUE: (.*?) :END"
    match = re.search(pattern, value)
    if match:
        return match.group(1)
    return None
    
if __name__ == "__main__":
    
    from dotenv import load_dotenv
    load_dotenv()
    from src.python.dwight.tools.brave import brave_web_search
    from src.python.dwight.tools.jina import get_url_as_text
    user_text = "top sensex 100 stocks by market capitalization"
    url_list = brave_web_search(os.getenv("BRAVE_API_KEY"), "top sensex 100 stocks by market capitalization")
    web_text = get_url_as_text(os.getenv("JINA_API_KEY"), url_list[0]['url'])
    output = generate_json_array_from_text(GOOGLE_API_KEY, user_text, web_text)
    df = convert_list_row_to_dataframe(output)
    print(df.head())
    