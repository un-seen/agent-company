import requests
from typing import Dict

# Literal type

from typing import Literal

METHOD = Literal["graphql", "sql"]

class SurrealExecutor:
    """
    A local surreal graphql executor that can perform HTTP requests.
    
    This environment takes as input a graphql query and 
    executes it using the requests library on the local surrealdb instance.
    """
    def __init__(self, base_url: str, namespace: str, database: str):
        """
        Initialize the environment with an optional base URL.
        
        Args:
            base_url (str): A base URL to prepend to every endpoint.
        """
        self.base_url = base_url
        self.auth = ("root", "root")
        self.namespace = namespace
        self.database = database
        
    
    def __call__(self, query: str, additional_variables: Dict, endpoint: METHOD):
        if endpoint == "graphql":
            headers = {
                "surreal-ns": self.namespace,
                "surreal-db": self.database,
                "Accept": "application/json"
            }
            payload = {
                "query": query
            }
            response = requests.post(f"{self.base_url}/{endpoint}", json=payload, headers=headers, auth=self.auth)
            return response.json()
        elif endpoint == "sql":
            headers = {
                "Accept": "application/json"
            }
            full_query = f"USE NS {self.namespace}; USE DB {self.database}; {query}"
            response = requests.post(f"{self.base_url}/{endpoint}", data=full_query, headers=headers, auth=self.auth)
            result = response.json()[2]["result"]
            return result
        
    


# Example usage:
if __name__ == "__main__":
    # Create an instance of the REST API environment with a base URL
    graphql_executor = SurrealExecutor(base_url="http://127.0.0.1:8000", namespace="prod", database="tempus", endpoint="graphql")
    query = "{ patient { id } }"
    result = graphql_executor(query, {})
    print(result)
    
    
    # Create an instance of the REST API environment with a base URL
    sql_executor = SurrealExecutor(base_url="http://127.0.0.1:8000", namespace="prod", database="tempus", endpoint="sql")
    query = "INFO FOR DB"
    result = sql_executor(query, {})
    print(result)
    