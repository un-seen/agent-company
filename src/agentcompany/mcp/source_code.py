import sys
import os

def get_source_code() -> str:
    """
    This function iterates over all modules in sys.modules.
    For each module, if it has a __file__ attribute that points to a Python file
    located in the current working directory, it reads the file's contents and
    appends it to a result string. Finally, it prints the complete string.
    """
    cwd = os.getcwd()
    cwd: str = os.path.abspath(cwd)
    aggregated_code = ""
    for name, module in sys.modules.items():
        module_file = getattr(module, '__file__', None)
        # Check that module_file exists, is a .py file, and is located in the current working directory.
        if module_file and module_file.endswith('.py'):
            # Normalize the paths to handle differences in formatting.
            module_file: str = os.path.abspath(module_file)
            
            if module_file.startswith(cwd):
                try:
                    with open(module_file, 'r', encoding='utf-8') as f:
                        code = f.read()
                    # Append a header for clarity
                    aggregated_code += f"# --- Module: {name} ---\n" + code + "\n\n"
                except Exception as e:
                    print(f"Could not read module '{name}' from '{module_file}': {e}")
    
    return aggregated_code
