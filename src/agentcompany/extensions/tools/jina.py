import requests
import os


JINA_API_KEY = os.getenv("JINA_API_KEY")

def get_url_as_text(target_url: str, retain_images: str = "none") -> str:
    """
    Fetch content through Jina.ai's proxy service with specified headers.
    
    Args:
        api_token (str): Jina.ai API token (Bearer token)
        target_url (str): Full URL to fetch through the proxy
        retain_images (str): Value for X-Retain-Images header (default: "none")
        
    Returns:
        str: Raw response content from the proxy
        
    Raises:
        requests.HTTPError: If the HTTP request returns an error status
    """
    base_url = "https://r.jina.ai/"
    full_url = f"{base_url}{target_url}"
    
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Retain-Images": retain_images
    }
    
    response = requests.get(full_url, headers=headers)
    response.raise_for_status()  # Raise exception for HTTP errors
    return response.text


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import json
    load_dotenv()
    response = get_url_as_text(os.environ["JINA_API_KEY"], "https://www.bseindia.com/markets/equity/eqreports/topmarketcapitalization.aspx")
    print(response)