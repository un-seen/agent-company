import os
from dotenv import load_dotenv
from driver import OpenAIServerModel
from typing import List
from application.cohort_builder.toolkit import execute_cohort_builder, index_file

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

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

def index_files(model: OpenAIServerModel, file_paths: List[str]):
    """
    Dummy function to simulate indexing files into the datalake.
    Replace this with your actual logic.
    
    Parameters:
        file_paths (list): A list of file paths to index.
    """
    console = Console()
    console.print("\n[sky_blue1]Indexing files into the datalake...[/sky_blue1]")
    for file_path in file_paths:
        # Here you would include the actual indexing logic.
        index_file(model, file_path)
        console.print(f"[light_sky_blue1]Indexing file:[/light_sky_blue1] [dodger_blue1]{file_path.strip()}[/dodger_blue1]")
    console.print("[sky_blue1]Indexing complete.[/sky_blue1]")

def main():
    console = Console()
    model = OpenAIServerModel(model_id="gpt-4o-mini")

    # Generate ASCII art headline using pyfiglet
    ascii_art = pyfiglet.figlet_format("agentos CLI", font="slant")
    console.print(f"[sky_blue1]{ascii_art}[/sky_blue1]")

    # Optionally, display a subtitle panel below the ASCII art.
    console.print(Panel("Welcome to agentos CLI", style="sky_blue1"))

    # Define available commands; extend this dictionary with additional commands as needed.
    commands = {
        "1": "cohort_builder",
        "2": "index_files",
        # "3": "yet_another_command",
    }

    while True:
        # Display the menu with a gradient of light blue shades.
        console.print("\n[light_sky_blue1]Please choose a command by typing the corresponding number:[/light_sky_blue1]")
        for key, command in commands.items():
            console.print(f"[deep_sky_blue1]{key}.[/deep_sky_blue1] [dodger_blue1]{command}[/dodger_blue1]")
        console.print("[sky_blue1]0.[/sky_blue1] [dodger_blue1]Exit[/dodger_blue1]")

        choice = Prompt.ask("[light_sky_blue1]Enter your choice[/light_sky_blue1]", default="0")
        if choice == "0":
            console.print("[dodger_blue1]Exiting... Goodbye![/dodger_blue1]")
            break

        if choice not in commands:
            console.print("[deep_sky_blue1]Invalid choice. Please try again.[/deep_sky_blue1]")
            continue

        selected_command = commands[choice]
        console.print(f"\n[light_sky_blue1]Selected command:[/light_sky_blue1] [dodger_blue1]{selected_command}[/dodger_blue1]")

        if selected_command == "cohort_builder":
            # Use prompt_toolkit's prompt to allow in-line editing of the prompt text.
            prompt_text = pt_prompt("Enter your prompt: ", style=pt_style)
            console.print(f"\n[deep_sky_blue1]You entered:[/deep_sky_blue1] [dodger_blue1]{prompt_text}[/dodger_blue1]")

            # Confirm that the user wants to proceed.
            if not Confirm.ask("[light_sky_blue1]Do you want to proceed with this prompt?[/light_sky_blue1]", default=True):
                console.print("[deep_sky_blue1]Prompt cancelled. Returning to command selection.[/deep_sky_blue1]")
                continue
                
            console.print(f"\n[sky_blue1]Executing '{selected_command}' with your prompt...[/sky_blue1]")
            execute_cohort_builder(model, prompt_text)

        elif selected_command == "index_files":
            # Allow the user to enter one or more file paths, comma-separated.
            files_input = pt_prompt("Enter the file path(s) to index (comma-separated): ", style=pt_style)
            # Split the input into a list of file paths.
            file_paths = [f.strip() for f in files_input.split(",") if f.strip()]
            console.print(f"\n[deep_sky_blue1]You entered:[/deep_sky_blue1] [dodger_blue1]{file_paths}[/dodger_blue1]")

            # Confirm that the user wants to proceed.
            if not Confirm.ask("[light_sky_blue1]Do you want to proceed with indexing these files?[/light_sky_blue1]", default=True):
                console.print("[deep_sky_blue1]Operation cancelled. Returning to command selection.[/deep_sky_blue1]")
                continue

            console.print(f"\n[sky_blue1]Indexing files...[/sky_blue1]")
            index_files(model, file_paths)
        else:
            console.print(f"[deep_sky_blue1]Command '{selected_command}' not recognized.[/deep_sky_blue1]")

        # Pause before returning to the main menu.
        Prompt.ask("[light_sky_blue1]Press Enter to return to the main menu[/light_sky_blue1]", default="", show_default=False)

if __name__ == "__main__":
    main()
