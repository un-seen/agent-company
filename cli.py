from agentcompany.application.consultant import ConsultantApp
from rich.console import Console
from rich.prompt import Prompt
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



def main():
    company_name = "PlanLLC"
    import os
    from redis import Redis
    redis_client = Redis.from_url(os.environ["REDIS_URL"])
    pubsub = redis_client.pubsub()
    pubsub.subscribe(company_name)
    console = Console()
    ceo_app = ConsultantApp(company_name, sop=f"""
    Standard Operating Procedure for {company_name}:
    
    1. Define strategy for the task at hand.
    2. Reason about the strategy to come up with a plan.
    3. Execute the plan and create a final answer.
    4. Get feedback on the final answer.
    5. Repeat steps 1-4 if feedback is not positive.
    """)
    ceo_app.start()
    # Generate ASCII art headline using pyfiglet
    ascii_art = pyfiglet.figlet_format("AgentCompany CLI", font="slant")
    console.print(f"[sky_blue1]{ascii_art}[/sky_blue1]")

    # Optionally, display a subtitle panel below the ASCII art.
    console.print(Panel("Welcome to AgentCompany CLI", style="sky_blue1"))
    
    # Run the main event loop connecting ui agent with user, console agent and executive strategy agent with the model.
    # Plan Identifier
    while True:
        # Action Prompt
        task = Prompt.ask(f"[light_sky_blue1]What do you want from {company_name}?[/light_sky_blue1]")
        if len(task.strip()) == 0:
            console.print("[deep_sky_blue1]No task provided. Returning to command selection.[/deep_sky_blue1]")
            continue
        elif task.lower() == "exit":
            break
        user_action_prompt = f"""Task: {task}"""
        ceo_app.send_user_input(user_action_prompt)
        # Complete Prompt
        get_attempt = 0
        max_attempts = 3
        while True:
            message = pubsub.get_message()
            if message is not None:
                Console().print(message)
                continue
            get_attempt += 1
            if get_attempt >= max_attempts:
                console.print(f"[red] {get_attempt} attempts made. Maximum attempts {max_attempts} reached. Exiting.[/red]")
                break
            
        user_action_prompt = Prompt.ask("[light_sky_blue1]Do you want to exit? (continues by default, any input exits the application)...[/light_sky_blue1]", default="no")
        if user_action_prompt.lower() == "no":
            continue
        else:
            break

if __name__ == "__main__":
    main()
