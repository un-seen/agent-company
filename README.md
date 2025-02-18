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

## Features

- **Modular Agent Architecture:** Build custom agent workflows by combining specialized agents.
- **Dynamic Tool Integration:** Easily add and validate tools (e.g., code execution, web search, memory lookup).
- **Real-Time Monitoring:** Utilize rich logging and monitoring tools to visualize agent activity.
- **Flexible Executors:** Run local Bash scripts or query a SurrealDB instance using GraphQL or SQL.
- **Extensible Framework:** Designed to allow integration with third-party services, additional agent types, and custom workflows.

## Installation

### Prerequisites

- **Python 3.8+**
- [Redis](https://redis.io/) – used for messaging and task queues.
- Environment packages for optional features (e.g., `duckduckgo-search`, `dotenv`, `rich`, `logfire`).

### Setup

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

### Usage

### Running the Consultant Application

The ConsultantApp serves as an example of how to set up and run an agent-driven application.

1. **Start the Application:**
    Write a simple runner script (e.g., run_consultant.py) with the following:
    ```python
    from agentcompany.application.consultant import ConsultantApp

    # Initialize the application with your company name and an optional SOP.
    app = ConsultantApp(company_name="YourCompany")
    app.start()

    # Send a sample user input (which will be processed by the manager agent).
    app.send_user_input("Initiate task: generate a strategic plan.")
    ```
    You can extend this file with additional configuration as needed.

2. **Monitor the Console:**
    The agent will start a background worker thread that listens to Redis, processes messages, and logs output using rich formatting.

### Extending the framework

- Adding Tools: Extend the tool set by creating new tool classes following the patterns in driver/default_tools.py.
- Creating Agents: Build your own ManagedAgents by extending the base classes in driver/agent_app.py.
- Integration: Use executors like LocalBashInterpreter or SurrealExecutor for specific tasks or integrations.

### Project Structure
```graphql
agent-company/
├── src/
│   └── agentcompany/
│       ├── application/
│       │   └── consultant/
│       │       ├── __init__.py          # Exposes ConsultantApp
│       │       └── create.py            # Implements ConsultantApp and default SOP
│       ├── driver/
│       │   ├── __init__.py              # Core imports and version info
│       │   ├── agent_app.py             # Defines BasicApp and worker thread logic
│       │   ├── default_tools.py         # Includes PythonInterpreterTool, DuckDuckGoSearchTool, etc.
│       │   ├── local_bash_executor.py   # Safe Bash command execution
│       │   ├── monitoring.py            # Logging and monitoring utilities using rich
│       │   ├──
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


### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more details.

## Acknowledgements
Developed with ❤️ by [@aloobhujiyan](https://twitter.com/aloobhujiyan)
Thanks to the open-source community for encouraging me to start this.
Agent Company is an evolving project—your feedback and contributions are crucial to its growth. Happy coding!

Copy
