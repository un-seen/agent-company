import requests
from typing import TypedDict


BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

class BraveSearchResult(TypedDict):
    title: str
    url: str
    source: str
    favicon: str


def brave_web_search(query: str) -> list[BraveSearchResult]:
    """
    Perform a web search using the Brave Search API.
    
    Args:
        api_key (str): Your Brave API subscription token.
        
    Returns:
        dict: JSON response containing search results.
        
    Raises:
        requests.HTTPError: If the HTTP request returns an error status.
    """
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {"q": query}
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()  # Raise exception for HTTP errors
    results = response.json()['web']['results']
    data = []
    for item in results:
        data.append({
            "title": item['title'],
            "url": item['url'],
            "source": item['profile']["name"],
            "favicon": item['profile']["img"]
        })
    return data


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import json
    load_dotenv()
    response = brave_web_search(os.environ["BRAVE_API_KEY"], "top sensex 100 stocks by market capitalization")['web']['results']
    data = []
    for item in response:
        data.append({
            "title": item['title'],
            "url": item['url'],
            "source": item['profile']["name"],
            "favicon": item['profile']["img"]
        })
    
    print(json.dumps(data, indent=2))