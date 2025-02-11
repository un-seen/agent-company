import subprocess
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_cli():
    """
    Run the CLI app using uv and return the CompletedProcess.
    """
    result = subprocess.run(
        ["uv", "run", "cli.py"],
        capture_output=True,
        text=True
    )
    return result

def add_dependency(dep):
    """
    Add a dependency using 'uv add' and install all requirements.
    """
    print(f"Adding missing dependency: {dep}")
    # Add the dependency (this should update your requirements file)
    add_result = subprocess.run(["uv", "add", dep], capture_output=True, text=True)
    if add_result.returncode != 0:
        print(f"Error adding dependency {dep}:\n{add_result.stderr}")
    else:
        print(f"Successfully added {dep}. Installing dependencies...")
        # Install dependencies from requirements.txt using uv pip.
        install_result = subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"], capture_output=True, text=True)
        if install_result.returncode != 0:
            print(f"Error installing dependencies:\n{install_result.stderr}")
        else:
            print("Dependencies installed successfully.")

def main():
    """
    Main loop: run the CLI app, check for missing dependency errors,
    add the dependency and install if needed, and then retry.
    """
    while True:
        print("Running CLI app via uv run cli.py ...")
        result = run_cli()
        # Print stdout and stderr for debugging.
        print(result.stdout)
        print(result.stderr)
        
        if result.returncode == 0:
            print("cli.py ran successfully.")
            break

        # Check stderr for a "No module named '...'" error.
        missing_dep_match = re.search(r"No module named '([^']+)'", result.stderr)
        if missing_dep_match:
            missing_dep = missing_dep_match.group(1)
            add_dependency(missing_dep)
            print("Retrying cli.py...\n")
        else:
            print("Error encountered but no missing dependency was detected.")
            break

if __name__ == '__main__':
    main()
