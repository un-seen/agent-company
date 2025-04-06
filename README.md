# Agent Company

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/un-seen/agent-company/ci.yml?branch=main)](https://github.com/un-seen/agent-company/actions)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)


_A modular framework for building autonomous agents._

Agent Company is designed to help developers build sophisticated multi-agent systems. The framework leverages Redis for messaging and provides a flexible API to incorporate custom tools and agents.

## Overview

Key components include:
- **Agent** 
- **Driver** 
- **Environment** 
- **Protocol** 

## 🌟 Features

<div align="center">

#### Environments

The following table organizes the environment types, providing a structured view for clarity:

| Category | Subcategory | Description | Example |
|----------|-------------|-------------|---------|
| Observability | Fully Observable | The agent has complete information about the state of the environment at any time, requiring no history tracking. | Chess, where the board and moves are fully visible. |
|          | Partially Observable | The agent has incomplete information, needing to infer hidden states. | Driving, where road conditions beyond corners are unknown. |
| Determinism | Deterministic | The next state is completely determined by the current state and action, with no randomness. | Chess, where each move has a definite outcome. |
|          | Stochastic | The next state involves randomness, not fully predictable by the agent. | Self-driving cars, affected by unpredictable driver behaviors. |
| Interaction Type | Competitive | Agents compete to optimize their own objectives, often in zero-sum games. | Chess, where players aim to defeat each other. |
|          | Cooperative | Agents work together to achieve a common goal, coordinating actions. | Multiple self-driving cars cooperating to avoid collisions, as noted in [AITUDE Understand Types of Environments](https://www.aitude.com/understand-types-of-environments-in-artificial-intelligence/). |
| Number of Agents | Single-agent | Only one agent operates in the environment, focusing on its task. | A person navigating a maze alone. |
|          | Multi-agent | Multiple agents interact, potentially competing or cooperating. | A game of football with 11 players per team, as per [Tpoint Tech Agent Environment](https://www.tpointtech.com/agent-environment-in-ai). |
| Environment Dynamics | Static | The environment remains unchanged over time, except by agent actions. | An empty house, where entering doesn't alter surroundings. |
|          | Dynamic | The environment changes independently of agent actions over time. | A roller coaster ride, constantly in motion, as described in [Applied AI Course Types of Environment](https://www.appliedaicourse.com/blog/types-of-environment-in-ai/). |
| State and Action Space | Discrete | The environment has a finite number of states or actions, countable and distinct. | Chess, with a finite set of possible moves per game. |
|          | Continuous | The environment has an infinite number of states or actions, not countable. | Self-driving cars, with continuous driving and parking actions. |
| Task Structure | Episodic | The agent's actions are divided into independent episodes, each with no dependency on previous ones. | A pick-and-place robot inspecting parts on a conveyor belt, deciding per part, as per [CSVeda AI Environment Types](https://csveda.com/ai-environment-types/). |
|          | Sequential | The agent's actions are interdependent, with previous decisions affecting future ones. | Checkers, where each move impacts subsequent strategies. |
| Environment Knowledge | Known | The agent knows the complete model, including how actions lead to state transitions. | Chess, with known rules and outcomes for moves. |
|          | Unknown | The agent does not know the model, requiring learning through interaction. | An agent in an unexplored environment, needing to discover dynamics, as noted in [Slideshare Types of Environment in AI](https://www.slideshare.net/slideshow/types-of-environment-in-artificial-intelligence/125168345). |

</div>

- 🚀 **Coming Soon**: Remote MCP Server Support!
- 🔍 Integrated memory search using Redis
- 🤝 OpenAI-compatible API endpoints
- 📖 Finetuning, Cloud SFT & GRPO training support for base LLMs


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
