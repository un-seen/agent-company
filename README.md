# Agent Company

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/un-seen/agent-company/ci.yml?branch=main)](https://github.com/un-seen/agent-company/actions)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)


_A modular framework for building and orchestrating autonomous agent applications._

Agent Company is designed to help developers build sophisticated multi-agent systems. With a modular architecture, it provides tools and abstractions to manage agents, execute code dynamically, search and manage memory, and even execute shell commands safely. The framework leverages Redis for messaging, integrates real-time logging with rich and logfire, and provides a flexible API to incorporate custom tools and agents.

## Overview

At its core, Agent Company is built around the idea of decomposing tasks into specialized agents. For example, the provided **ConsultantApp** in the `application/consultant` module demonstrates how to create a ManagerAgent that orchestrates multiple ManagedAgents – a reasoning specialist and a plan specialist – each empowered by dedicated Python code agents and custom tools such as memory search.

Key components include:
- **Agent Applications:** Abstract base classes and implementations (e.g., `BasicApp` and `ConsultantApp`) to kickstart agent orchestration.
- **Drivers:** Core modules for agents, tools, monitoring, and memory management. This includes tools like the Python interpreter, DuckDuckGo search, and user input handling.
- **Executors:** Components for running external commands such as a safe Bash interpreter and a SurrealDB executor for GraphQL/SQL queries.
- **Validation & Types:** Built-in mechanisms for tool validation and type handling to ensure that agent outputs (text, images, audio) are correctly processed and displayed.

## 🌟 Features

<div align="center">

| 🧩 Modular Architecture | 🔧 Dynamic Tools | 📊 Real-time Monitoring |
|-------------------------|------------------|-------------------------|
| Build agent hierarchies | Add custom tools | Rich logging dashboard  |

</div>

- 🚀 **Coming Soon**: SFT & GRPO support for base LLMs!
- 🛡️ Safe shell execution with permission controls
- 🔍 Integrated memory search using Redis
- 🤝 OpenAI-compatible API endpoints

## Dependencies

- **Python 3.8+**
- [Redis](https://redis.io/) – used for memory and logging.
- [Logfire](https://logfire.pydantic.dev/) – used for monitoring.

## Setup

1. **Clone the Repository:**

    ```bash
    git clone https://gitingest.com/un-seen/agent-company.git
    cd agent-company/src/agentcompany
    ```

2. **Create a Virtual Environment and Install Dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3. **Configure Environment Variables:**
    Create a .env file in the repository root with at least the following variables:
    ```bash
    REDIS_URL=redis://localhost:6379
    LOGFIRE_TOKEN=your_logfire_token  # Optional logging token
    ```
    You can extend this file with additional configuration as needed.

## Usage

### Building and Running a Custom Agent Application

The ConsultantApp demonstrates how to extend the BasicApp abstract class to create a custom agent application. Below is an example of how to set up and run the ConsultantApp with user input.

#### Example: ConsultantApp
```py
from agentcompany.framework import BasicApp
from agentcompany.driver import PythonCodeAgent, ManagerAgent

class StartupConsultants(BasicApp):
    def create_manager_agent(self):
        developer = PythonCodeAgent(
            name="fullstack_dev",
            tools=[web_scraper, sql_executor],
            description="Full-stack development expert"
        )
        
        strategist = PythonCodeAgent(
            name="growth_hacker",
            tools=[market_analyzer, seo_optimizer],
            description="Digital growth specialist"
        )
        
        return ManagerAgent(
            managed_agents=[developer, strategist],
            name="startup_ceo"
        )

# Launch your agent team
app = StartupConsultants(company_name="TechPioneers")
app.send_user_input("Develop MVP for AI-powered analytics platform")
```

## Project Structure
```graphql
agent-company/
├── src/
│   └── agentcompany/
│       ├── driver/
│       │   ├── __init__.py              # Core imports and version info
│       │   ├── agent_app.py             # Defines BasicApp and worker thread logic
│       │   ├── default_tools.py         # Includes PythonInterpreterTool, DuckDuckGoSearchTool, etc.
│       │   ├── local_bash_executor.py   # Safe Bash command execution
│       │   ├── monitoring.py            # Logging and monitoring utilities using rich
│       │   ├──
│   └── lib/agents
│       │   ├── __init__.py              # Core imports and version info
│       │   ├── bash.py                  # Write bash code and runs it in local bash environment
│       │   ├── graphql.py               # Writes graphql code and interfaces with surrealdb 
│       │   ├── psql.py                  # Writes sql code and interfaces with postgresql
│       │   ├── tfserving.py             # Writes json blobs to call tfserving agents
```

## Contribute

Contributions are very welcome! To contribute:

1. **Fork the Repository:**
2. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Make Your Changes and Commit::**
   ```bash
   git checkout -b feature/YourFeature
   ```
4. **Push Your Branch:**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request:**

Please follow our contribution guidelines and code style. For more details, refer to the CONTRIBUTING.md file.


## Acknowledgements
Developed with ❤️ by [@aloobhujiyan](https://twitter.com/aloobhujiyan)
Thanks to the open-source community for encouraging me to start this.
Agent Company is an evolving project—your feedback and contributions are crucial to its growth. Happy coding!
