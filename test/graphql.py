from agentcompany.llms.base import OpenAIServerModel
from agentcompany.lib.agents.graphql import GraphqlAgent

if __name__ == "__main__":
    
    model = OpenAIServerModel("gpt-4o-mini")
    base_url="http://127.0.0.1:8000"
    namespace="prod"
    database="tempus"
    agent = GraphqlAgent(
        model=model,
        base_url=base_url,
        namespace=namespace,
        database=database,
    )
    agent.run("Fetch patients with cancer recurrence")