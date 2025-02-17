from agentcompany.driver import PythonCodeAgent, ManagerAgent, OpenAIServerModel, tool, ManagedAgent
from typing import List, Generator, Dict
from rich import console
import time
import rich
from agentcompany.driver.memory import get_memory_index_name

from dotenv import load_dotenv
# Load environment variables
load_dotenv()


pc, agent_index_name = get_memory_index_name()

if not pc.has_index(name=agent_index_name):
    from pinecone import ServerlessSpec
    pc.create_index(
            name=agent_index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            ) 
        ) 
        
    while not pc.describe_index(agent_index_name).status['ready']:
        time.sleep(1)
            
@tool
def memory_search(query: str) -> List[str]:
    """
    It searches for relevant information in the past memories or new discoveries.
    Discoveries are immediately stored in the memory.
    Args:
        query: The text query to search for in the memory
    Returns:
        results: a list of the memories to the query
    """
    # Convert the text into numerical vectors that Pinecone can index
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    # Search the index for the three most similar vectors
    index = pc.Index(agent_index_name)
    results = index.query(
        namespace="prod",
        vector=query_embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    rich.console.Console().print(f"Results: {results}")
    
    if len(results["matches"]) == 0:
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
            ) from e
        ddgs = DDGS()
        results = ddgs.text(query, max_results=3)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return postprocessed_results
    return results["matches"]


@tool
def invest_in_btc(amount: float) -> bool:
    """
    It invests in bitcoin with the given amount.
    Args:
        amount: The amount to invest in bitcoin
    Returns:
        success: a boolean indicating if the investment was successful
    """
    return amount > 0

# @tool
# def graph_search_tool(path: str) -> str:
#     """
#     It loads csv data into a pandas dataframe
#     Args:
#         path: The path to the data for the csv
#     Returns:
#         df: a pandas dataframe
#     """
#     Use redis or mongo
#     redis_client = redis_connect()
#     system_client = mongodb_connect()
#     system_state = system_client["agentos"]
#     collection = system_state["datalake"]
#     document = collection.find_one({"_id": path})
#     file_dictionary = FileDictionary.model_validate(document)
#     return file_dictionary.to_dataframe()
    
@tool
def console_tool(content: str) -> None:
    """
    It displays the content in the console
    Args:
        content: The string content to display in the console
    Returns:
        None
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.text import Text

    console.Console().print(Group(
        Rule(
            "[italic]UI:",
            align="left",
            style="orange",
        ),
        Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        ),
    ))


def run_with(model: OpenAIServerModel, company_name: str, sop: str) -> ManagerAgent:
    # REASONING SPECIALIST
    managed_reasoning_agent = ManagedAgent(
        company_name=company_name,
        agent=PythonCodeAgent(
            name="reasoningspecialist",
            managed_agents=[],
            tools=[memory_search],
            model=model,
            additional_authorized_imports=[],
            step_callbacks=[],
            max_steps=3,
        ),
        description=f"""Given a step of actions and the user objective, it reasons about the prompt to come up with parallel possible plan of actions with slight suggestive modifications if necessary to complete the objective."""
    )
    # PLAN SPECIALIST
    managed_plan_agent = ManagedAgent(
        company_name=company_name,
        agent=PythonCodeAgent(
            name="planspecialist",
            managed_agents=[], # managed_dream_agent, managed_subconscious_agent],
            tools=[],
            model=model,
            additional_authorized_imports=[],
            step_callbacks=[],
            max_steps=3,
            verbosity_level=2,
        ),
        description=f"""Given an objective guidance, it creates a plan of actions to complete the objective.""",
    )
    # MANAGER
    ceo_agent = ManagerAgent(
        company_name=company_name,
        sop=sop,
        model=model,
        managed_agents=[managed_plan_agent, managed_reasoning_agent],
        additional_authorized_imports=["*"],
        step_callbacks=[],
        max_steps=3,
        verbosity_level=2,
        name="ceo",
    )
    return ceo_agent
    