import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_MAX_LEN_OUTPUT = 5000

class LocalBashInterpreter:
    """
    A simple Bash interpreter class that takes as input:
      - a list of Bash scripts (as strings) that define tools (helper functions, aliases, etc.)
      - a code_action string (Bash code) that can use these tools.
    
    The interpreter builds a temporary script that first sources all the tool scripts
    and then executes the code_action.
    
    **Warning:** This is a simplified proof‑of‑concept. Sandboxing shell code requires
    additional security measures.
    """
    
    def __init__(
        self,
        bash_tool_scripts: Optional[Dict[str, str]] = None,
        allowed_commands: Optional[List[str]] = None,
        max_print_outputs_length: Optional[int] = None,
    ):
        # List of Bash tool scripts to be sourced before executing code_action.
        self.tool_scripts = [v for k,v in bash_tool_scripts.items()] if bash_tool_scripts is not None else []
        
        # Define a whitelist of allowed Bash commands.
        if allowed_commands is None:
            self.allowed_commands = [
                "echo", "ls", "pwd", "cat", "grep", "awk",
                "sed", "cut", "date", "set", "env"
            ]    
        
        self.allowed_commands.extend([k for k in bash_tool_scripts.keys()])

        self.state: Dict[str, Any] = {}  # Additional state variables (if needed)
        self.max_print_outputs_length = max_print_outputs_length or DEFAULT_MAX_LEN_OUTPUT
        self.print_outputs = ""  # Log buffer for all command outputs

    def _is_command_allowed(self, code: str) -> bool:
        """
        Naively check if the first token of the Bash code (after tool scripts are sourced)
        is in our whitelist.
        """
        code = code.strip()
        if not code:
            return True
        # Extract the first token of the user's code_action.
        first_token = code.split()[0]
        return first_token in self.allowed_commands

    def __call__(self, code_action: str, additional_variables: Dict[str, Any] = None) -> Tuple[str, str, bool]:
        """
        Executes the given Bash code string (code_action) after first sourcing the tool scripts.

        Args:
            code_action (str): The Bash code to execute.
            additional_variables (Dict[str, Any], optional): Additional environment variables
                to set for the subprocess.

        Returns:
            Tuple[str, str, bool]:
              - The combined output (stdout and stderr),
              - The accumulated log of outputs,
              - A boolean flag is_final_answer (always False for this interpreter).
        """
        # Prepare the environment variables.
        env = os.environ.copy()
        if additional_variables:
            for key, value in additional_variables.items():
                # Convert values to strings for the environment.
                env[key] = str(value)

        # Check that the user-provided code_action is allowed.
        if not self._is_command_allowed(code_action):
            raise ValueError(f"Command '{code_action.split()[0]}' is not allowed by the safe interpreter.")

        # Build the full Bash script to execute.
        # First, we source (or simply include) all tool scripts.
        # Then we run the code_action.
        full_script = "\n".join(self.tool_scripts) + "\n" + code_action

        # Execute the full script using Bash.
        result = subprocess.run(
            ["bash", "-c", full_script],
            capture_output=True,
            text=True,
            env=env,
        )

        # Combine stdout and stderr.
        output = result.stdout + result.stderr

        # Append to our log buffer (truncate if necessary).
        self.print_outputs += output + "\n"
        if len(self.print_outputs) > self.max_print_outputs_length:
            self.print_outputs = self.print_outputs[-self.max_print_outputs_length:]

        # In this simple interpreter, we do not have a 'final_answer' concept.
        is_final_answer = False

        return output, self.print_outputs, is_final_answer


# Example usage:
if __name__ == "__main__":
    # Define a sample tool script that creates a helper function.
    tool_script_name = "greet"
    tool_script = r'''
    #!/bin/bash
    # This tool creates a helper function "greet" that prints a greeting.
    function greet() {
      echo "Hello, $1! Welcome to the safe Bash interpreter."
    }
    '''
    # Create an interpreter with the tool script.
    interpreter = LocalBashInterpreter(bash_tool_scripts={
        tool_script_name: tool_script
    })
    
    # Sample code action that uses the tool "greet"
    bash_code = 'greet "Alice"; ls -l'
    
    try:
        output, logs, is_final = interpreter(bash_code)
        print("Output:")
        print(output)
        print("Logs:")
        print(logs)
    except ValueError as e:
        print(f"Error: {e}")

