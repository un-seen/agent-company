import os
from agentcompany.driver import PythonCodeAgent, ManagerAgent, ManagedAgent
from agentcompany.driver.agent_app import BasicApp
import logfire
from typing import Any
from agentcompany.driver import PythonCodeAgent, ManagerAgent, tool, ManagedAgent
from typing import List, Generator, Dict
import time
import rich
from agentcompany.driver.memory import get_memory_index_name
from dotenv import load_dotenv

if os.environ.get("LOGFIRE_TOKEN", None):
    logfire.configure(token=os.environ["LOGFIRE_TOKEN"])


def default_sop(company_name: str) -> str:
    return f"""
    Standard Operating Procedure for {company_name}:
    
    1. Define strategy for the task at hand.
    2. Reason about the strategy to come up with a plan.
    3. Execute the plan and create a final answer.
    4. Get feedback on the final answer.
    5. Repeat steps 1-4 if feedback is not positive.
    """
    
    
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
    load_dotenv()


    agent_index_name = get_memory_index_name()

    from redis import Redis
    import os
    redis_client = Redis.from_url(os.environ["REDIS_URL"])
    all_memory = redis_client.get(agent_index_name)
    rich.console.Console().print(f"Memory: {all_memory}")
    if len(all_memory) == 0:
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


class ConsultantApp(BasicApp):
    
    def __init__(self, company_name, sop: str = None, model_name = "gpt-4o-mini", **kwargs):
        sop = sop or default_sop(company_name)
        super().__init__(company_name, sop, model_name, **kwargs)
        
    def create_manager_agent(self) -> ManagerAgent:
        """
        Implements the abstract method by creating a ManagerAgent composed of
        a reasoning specialist and a plan specialist.
        """
        company_name = self.company_name
        sop = self.sop
        model = self.model

        # Create a managed reasoning agent.
        managed_reasoning_agent = ManagedAgent(
            company_name=company_name,
            agent=PythonCodeAgent(
                name="reasoningspecialist",
                managed_agents=[],
                tools=[],
                model=model,
                additional_authorized_imports=[],
                step_callbacks=[],
                max_steps=3,
            ),
            description=(
                f"Given a step of actions and the user objective, "
                f"it reasons about the prompt to come up with parallel possible plans of actions, "
                f"with slight suggestive modifications if necessary to complete the objective."
            )
        )

        # Create a managed plan agent.
        managed_plan_agent = ManagedAgent(
            company_name=company_name,
            agent=PythonCodeAgent(
                name="planspecialist",
                managed_agents=[],
                tools=[memory_search],
                model=model,
                additional_authorized_imports=[],
                step_callbacks=[],
                max_steps=3,
                verbosity_level=2,
            ),
            description=(
                f"Given objective guidance, it creates a plan of actions to complete the objective."
            )
        )

        # Create the CEO ManagerAgent that uses the managed agents.
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
