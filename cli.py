import os
from dotenv import load_dotenv
from driver import OpenAIServerModel
from typing import List
from driver import CodeAgent
from application.ui.toolkit import run_with as run_with_ui
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
import random

# Import prompt_toolkit's prompt for an editable input experience.
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.styles import Style

# Import pyfiglet for ASCII art
import pyfiglet

# Define a style for the prompt_toolkit input (using a light blue shade).
pt_style = Style.from_dict({
    '': '#87CEFA',  # LightSkyBlue color for the text
})

# Load environment variables
load_dotenv()

def main():
    console = Console()
    model = OpenAIServerModel(model_id="gpt-4o-mini")
    
    # Generate ASCII art headline using pyfiglet
    ascii_art = pyfiglet.figlet_format("AgentCompany CLI", font="slant")
    console.print(f"[sky_blue1]{ascii_art}[/sky_blue1]")

    # Optionally, display a subtitle panel below the ASCII art.
    console.print(Panel("Welcome to AgentCompany CLI", style="sky_blue1"))
    
    
    # Run the main event loop connecting ui agent with user, console agent and executive strategy agent with the model.
    ui_agent: CodeAgent = run_with_ui(model)
    while True:
        
        # Plan Identifier
        plan_id = str(random.randint(1000, 9999))
        # Action Prompt
        objective_guidance = Prompt.ask("[light_sky_blue1]What do you from Agent Company?[/light_sky_blue1]")
        if len(objective_guidance.strip()) == 0:
            console.print("[deep_sky_blue1]No action provided. Returning to command selection.[/deep_sky_blue1]")
            continue
        elif objective_guidance.lower() == "exit":
            break
        user_action_prompt = f"""Plan id = {plan_id} | Objective guidance from user = {objective_guidance}"""
        # Complete Prompt
        while True:
            ui_agent.run(user_action_prompt)
            user_action_prompt = Prompt.ask("[light_sky_blue1]You...[/light_sky_blue1]", default="exit")
            if user_action_prompt.lower() == "exit":
                break

if __name__ == "__main__":
    main()
