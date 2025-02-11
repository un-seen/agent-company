from driver import CodeAgent, OpenAIServerModel, tool, ManagedAgent
from driver.agents import ActionStep
import pandas as pd
from typing import List, Generator, Dict
from PIL import Image
import os
import numpy as np
from rich import console
from upstash_redis import Redis
import time
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import rich
from driver.memory import get_pinecone_client

from dotenv import load_dotenv
# Load environment variables
load_dotenv()


pc, agent_index_name = get_pinecone_client()

if not pc.has_index(name=agent_index_name):
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
def search(query: str) -> List[str]:
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
    
    
@tool
def start_plan_tool(plan_id: str) -> None:
    """
    It stores the plan id in memory and starts the plan
    Args:
        plan_id: The plan id to start
    Returns:
        None
    """
    redis_client = Redis.from_env()
    plan_key = f"plan:{plan_id}"
    redis_client.set(plan_key, "started")
    
@tool
def complete_plan_tool(plan_id: str) -> None:
    """
    It stores the plan id in memory and completes the plan
    Args:
        plan_id: The plan id to complete
    Returns:
        None
    """
    redis_client = Redis.from_env()
    plan_key = f"plan:{plan_id}"
    redis_client.set(plan_key, "completed")
    
    
@tool
def get_objective_tool(plan_id: str) -> str:
    """
    It retrieves the objective from memory
    Args:
        plan_id: The plan id for which to fetch objective.
    Returns:
        A string representing the objective.
    """
    redis_client = Redis.from_env()
    plan_key = f"plan_objective:{plan_id}"
    return redis_client.get(plan_key)


@tool
def retreive_user_input_tool(plan_id: str, request_id: str, prompt: str) -> str:
    """
    It retrieves the user input from the console
    Args:
        plan_id: The plan id for which to user input
        request_id: A request id to store the user input
        prompt: The prompt to display to the user
    Returns:
        A string representing the user input.
    """
    value = input(prompt)
    redis_client = Redis.from_env()
    user_input_key = f"plan_id:{plan_id}/user_input:{request_id}"
    redis_client.set(user_input_key, value)
    print(f"User input for prompt '{prompt}': {value}")
    return value

@tool 
def fetch_user_input_tool(plan_id: str, request_id: str) -> str:
    """
    It retrieves the user input from the console
    Args:
        plan_id: The plan id for which to get user input
        request_id: A request id to store the user input
    Returns:
        A string representing the user input.
    """
    redis_client = Redis.from_env()
    user_input_key = f"plan_id:{plan_id}/user_input:{request_id}"
    return redis_client.get(user_input_key)
    
@tool
def set_objective_tool(plan_id: str, objective: str) -> None:
    """
    It stores the objective in memory
    Args:
        plan_id: The plan id for which to store objective.
        objective: The objective to store.
    Returns:
        None
    """
    redis_client = Redis.from_env()
    plan_key = f"plan_objective:{plan_id}"
    redis_client.set(plan_key, objective)


def run_with(model: OpenAIServerModel) -> CodeAgent:
    
    codespecialist = CodeAgent(
        name="codespecialist",
        managed_agents=[],
        tools=[console_tool, retreive_user_input_tool, fetch_user_input_tool, search],
        model=model,
        additional_authorized_imports=["*"],
        step_callbacks=[],
        max_steps=3,
    )
    
    managed_code_agent = ManagedAgent(
        agent=codespecialist,
        description=f"""Given an objective guidance, it writes code to act on the python environment to make progress in the objective."""   
    )
    
    reasoningspecialist = CodeAgent(
        name="reasoningspecialist",
        managed_agents=[],
        tools=[console_tool, retreive_user_input_tool, fetch_user_input_tool, search],
        model=model,
        additional_authorized_imports=[],
        step_callbacks=[],
        max_steps=3,
    )
    
    managed_reasoning_agent = ManagedAgent(
        agent=reasoningspecialist,
        description=f"""Given a step of actions and the user objective, it reasons about the prompt to come up with parallel possible plan of actions with slight suggestive modifications if necessary to complete the objective."""
    )

    planspecialist = CodeAgent(
        name="planspecialist",
        managed_agents=[], # managed_dream_agent, managed_subconscious_agent],
        tools=[console_tool, retreive_user_input_tool, fetch_user_input_tool, search],
        model=model,
        additional_authorized_imports=[],
        step_callbacks=[],
        max_steps=3,
        verbosity_level=2,
    )
    
    # A genetic memory can be any piece of text or url that the user has seen before. 
    # The PlanAgent only job is to think of the best plan to service the user and
    # then call the LLM with the user prompt and next step of the plan.
    managed_plan_agent = ManagedAgent(
            agent=planspecialist,
            description=f"""Given an objective guidance, it creates a plan of actions to complete the objective.""",
        )
        
    # managed_plan_agent(request=request)  
    
    ui_agent = CodeAgent(
        name="agentisui",
        managed_agents=[managed_plan_agent, managed_reasoning_agent, managed_code_agent],
        tools=[console_tool, retreive_user_input_tool, fetch_user_input_tool, search, invest_in_btc],
        model=model,
        additional_authorized_imports=["*"],
        step_callbacks=[],
        max_steps=3,
        verbosity_level=2,
    )
    
    return ui_agent
    