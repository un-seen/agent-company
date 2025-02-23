from agentcompany.driver.models import OpenAIServerModel
from agentcompany.lib.agents.psql import PsqlAgent

if __name__ == "__main__":
    
    model = OpenAIServerModel("gpt-4o-mini")
    pg_config = {
        "host": "127.0.0.1",
        "port": 5432,
        "dbname": "core",
        "user": "postgres",
        "password": "root"
    }
    agent = PsqlAgent(
        model=model,
        pg_config=pg_config,
    )
    data = agent.run("Identify patients who have a family history of cancer.")
    print(f"Result")
    print(str(data)[:100] + "...")