import ast
import re
import builtins
import difflib
import inspect
import math
import json
import sys
import re
import logging
from collections.abc import Mapping
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Union
import traceback
import subprocess
import numpy as np
import pandas as pd
from agentcompany.driver.markdown import json_to_markdown
from agentcompany.mcp.utils import truncate_content
from agentcompany.extensions.environments.base import ExecutionEnvironment
from agentcompany.mcp.base import ModelContextProtocolImpl
from agentcompany.extensions.environments.exceptions import InterpreterError, ReturnException, ERRORS, BreakException, ContinueException


logger = logging.getLogger(__name__)

BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]

_IMPORT_PACKAGE_MAP = {
    # Computer Vision
    "cv2": "opencv-python",
    
    # Web Scraping
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    
    # Data Science
    "sklearn": "scikit-learn",
    "pd": "pandas",
    "np": "numpy",
    "plt": "matplotlib",
    
    # Image Processing
    "PIL": "pillow",
    
    # Configuration
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    
    # Database
    "psycopg2": "psycopg2-binary",
    
    # Utilities
    "dateutil": "python-dateutil",
    "jwt": "pyjwt",
    "django": "django",
    "flask": "flask"
}

_STANDARD_LIBRARY_MODULES = {
    "sys", "os", "re", "json", "ast", "subprocess", "typing", "logging",
    "importlib", "collections", "datetime", "math", "random", "socket",
    "argparse", "itertools", "functools", "threading", "pathlib", "csv",
    "html", "http", "urllib", "xml", "email", "ssl", "hashlib", "base64",
    "io", "time", "unittest", "pdb", "traceback", "zipfile", "sqlite3",
    "glob", "pickle", "tempfile", "uuid", "webbrowser", "ctypes", "queue",
    "asyncio", "signal", "warnings", "weakref", "dataclasses", "enum",
    "statistics", "pprint", "textwrap", "shutil", "doctest", "profile"
}

PRINT_OUTPUTS = ""
OPERATIONS_COUNT, MAX_OPERATIONS = 0, 10000000

def custom_print(*args):
    return None

def get_pip_package(module_name: str) -> Optional[str]:
    """
    Maps Python import names to their corresponding PyPI package names.
    
    Args:
        module_name (str): The name used in import statements
        
    Returns:
        str: PyPI package name if mapping exists
        None: For standard library modules
        module_name: For unmapped packages (assumed same as import name)
    """
    # Check if it's a known standard library module
    if module_name in _STANDARD_LIBRARY_MODULES:
        return None
        
    # Return mapped package name if exists
    return _IMPORT_PACKAGE_MAP.get(module_name, module_name)

BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "isinstance": isinstance,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": math.pow,
    "sqrt": math.sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": getattr,
    "hasattr": hasattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "type": type,
    "complex": complex,
}


def get_iterable(obj):
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, "__iter__"):
        return list(obj)
    else:
        raise InterpreterError("Object is not iterable")


def fix_final_answer_code(code: str) -> str:
    """
    Sometimes an LLM can try to assign a variable to final_answer, which would break the final_answer() tool.
    This function fixes this behaviour by replacing variable assignments to final_answer with final_answer_variable,
    while preserving function calls to final_answer().
    """
    # First, find if there's a direct assignment to final_answer
    # Use word boundary and negative lookbehind to ensure it's not an object attribute
    assignment_pattern = r"(?<!\.)(?<!\w)\bfinal_answer\s*="
    if "final_answer(" not in code or not re.search(assignment_pattern, code):
        # If final_answer tool is not called in this blob, then doing the replacement is hazardous because it could false the model's memory for next steps.
        # Let's not modify the code and leave the subsequent assignment error happen.
        return code

    # Pattern for replacing variable assignments
    # Looks for 'final_answer' followed by '=' with optional whitespace
    # Negative lookbehind ensures we don't match object attributes
    assignment_regex = r"(?<!\.)(?<!\w)(\bfinal_answer)(\s*=)"
    code = re.sub(assignment_regex, r"final_answer_variable\2", code)

    # Pattern for replacing variable usage but not function calls
    # Negative lookahead (?!\s*\() ensures we don't match function calls
    # Negative lookbehind (?<!\.|\w) ensures we don't match object methods or other variables
    variable_regex = r"(?<!\.)(?<!\w)(\bfinal_answer\b)(?!\s*\()"
    code = re.sub(variable_regex, "final_answer_variable", code)
    return code


def evaluate_unaryop(
    expression: ast.UnaryOp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    operand = evaluate_ast(expression.operand, state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression.op, ast.USub):
        return -operand
    elif isinstance(expression.op, ast.UAdd):
        return operand
    elif isinstance(expression.op, ast.Not):
        return not operand
    elif isinstance(expression.op, ast.Invert):
        return ~operand
    else:
        raise InterpreterError(f"Unary operation {expression.op.__class__.__name__} is not supported.")


def evaluate_lambda(
    lambda_expression: ast.Lambda,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Callable:
    args = [arg.arg for arg in lambda_expression.args.args]

    def lambda_func(*values: Any) -> Any:
        new_state = state.copy()
        for arg, value in zip(args, values):
            new_state[arg] = value
        return evaluate_ast(
            lambda_expression.body,
            new_state,
            static_tools,
            custom_tools,
            authorized_imports,
        )

    return lambda_func


def evaluate_while(
    while_loop: ast.While,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    max_iterations = 1000
    iterations = 0
    while evaluate_ast(while_loop.test, state, static_tools, custom_tools, authorized_imports):
        for node in while_loop.body:
            try:
                evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
            except BreakException:
                return None
            except ContinueException:
                break
        iterations += 1
        if iterations > max_iterations:
            raise InterpreterError(f"Maximum number of {max_iterations} iterations in While loop exceeded")
    return None


def create_function(
    func_def: ast.FunctionDef,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Callable:
    def new_func(*args: Any, **kwargs: Any) -> Any:
        func_state = state.copy()
        arg_names = [arg.arg for arg in func_def.args.args]
        default_values = [
            evaluate_ast(d, state, static_tools, custom_tools, authorized_imports) for d in func_def.args.defaults
        ]

        # Apply default values
        defaults = dict(zip(arg_names[-len(default_values) :], default_values))

        # Set positional arguments
        for name, value in zip(arg_names, args):
            func_state[name] = value

        # Set keyword arguments
        for name, value in kwargs.items():
            func_state[name] = value

        # Handle variable arguments
        if func_def.args.vararg:
            vararg_name = func_def.args.vararg.arg
            func_state[vararg_name] = args

        if func_def.args.kwarg:
            kwarg_name = func_def.args.kwarg.arg
            func_state[kwarg_name] = kwargs

        # Set default values for arguments that were not provided
        for name, value in defaults.items():
            if name not in func_state:
                func_state[name] = value

        # Update function state with self and __class__
        if func_def.args.args and func_def.args.args[0].arg == "self":
            if args:
                func_state["self"] = args[0]
                func_state["__class__"] = args[0].__class__

        result = None
        try:
            for stmt in func_def.body:
                result = evaluate_ast(stmt, func_state, static_tools, custom_tools, authorized_imports)
        except ReturnException as e:
            result = e.value

        if func_def.name == "__init__":
            return None

        return result

    return new_func


def evaluate_function_def(
    func_def: ast.FunctionDef,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Callable:
    custom_tools[func_def.name] = create_function(func_def, state, static_tools, custom_tools, authorized_imports)
    return custom_tools[func_def.name]


def evaluate_class_def(
    class_def: ast.ClassDef,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> type:
    class_name = class_def.name
    bases = [evaluate_ast(base, state, static_tools, custom_tools, authorized_imports) for base in class_def.bases]
    class_dict = {}

    for stmt in class_def.body:
        if isinstance(stmt, ast.FunctionDef):
            class_dict[stmt.name] = evaluate_function_def(stmt, state, static_tools, custom_tools, authorized_imports)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    class_dict[target.id] = evaluate_ast(
                        stmt.value,
                        state,
                        static_tools,
                        custom_tools,
                        authorized_imports,
                    )
                elif isinstance(target, ast.Attribute):
                    class_dict[target.attr] = evaluate_ast(
                        stmt.value,
                        state,
                        static_tools,
                        custom_tools,
                        authorized_imports,
                    )
        else:
            raise InterpreterError(f"Unsupported statement in class body: {stmt.__class__.__name__}")

    new_class = type(class_name, tuple(bases), class_dict)
    state[class_name] = new_class
    return new_class


def evaluate_augassign(
    expression: ast.AugAssign,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    def get_current_value(target: ast.AST) -> Any:
        if isinstance(target, ast.Name):
            return state.get(target.id, 0)
        elif isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            return obj[key]
        elif isinstance(target, ast.Attribute):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            return getattr(obj, target.attr)
        elif isinstance(target, ast.Tuple):
            return tuple(get_current_value(elt) for elt in target.elts)
        elif isinstance(target, ast.List):
            return [get_current_value(elt) for elt in target.elts]
        else:
            raise InterpreterError("AugAssign not supported for {type(target)} targets.")

    current_value = get_current_value(expression.target)
    value_to_add = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)

    if isinstance(expression.op, ast.Add):
        if isinstance(current_value, list):
            if not isinstance(value_to_add, list):
                raise InterpreterError(f"Cannot add non-list value {value_to_add} to a list.")
            current_value += value_to_add
        else:
            current_value += value_to_add
    elif isinstance(expression.op, ast.Sub):
        current_value -= value_to_add
    elif isinstance(expression.op, ast.Mult):
        current_value *= value_to_add
    elif isinstance(expression.op, ast.Div):
        current_value /= value_to_add
    elif isinstance(expression.op, ast.Mod):
        current_value %= value_to_add
    elif isinstance(expression.op, ast.Pow):
        current_value **= value_to_add
    elif isinstance(expression.op, ast.FloorDiv):
        current_value //= value_to_add
    elif isinstance(expression.op, ast.BitAnd):
        current_value &= value_to_add
    elif isinstance(expression.op, ast.BitOr):
        current_value |= value_to_add
    elif isinstance(expression.op, ast.BitXor):
        current_value ^= value_to_add
    elif isinstance(expression.op, ast.LShift):
        current_value <<= value_to_add
    elif isinstance(expression.op, ast.RShift):
        current_value >>= value_to_add
    else:
        raise InterpreterError(f"Operation {type(expression.op).__name__} is not supported.")

    # Update the state: current_value has been updated in-place
    set_value(
        expression.target,
        current_value,
        state,
        static_tools,
        custom_tools,
        authorized_imports,
    )

    return current_value


def evaluate_boolop(
    node: ast.BoolOp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> bool:
    if isinstance(node.op, ast.And):
        for value in node.values:
            if not evaluate_ast(value, state, static_tools, custom_tools, authorized_imports):
                return False
        return True
    elif isinstance(node.op, ast.Or):
        for value in node.values:
            if evaluate_ast(value, state, static_tools, custom_tools, authorized_imports):
                return True
        return False


def evaluate_binop(
    binop: ast.BinOp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    # Recursively evaluate the left and right operands
    left_val = evaluate_ast(binop.left, state, static_tools, custom_tools, authorized_imports)
    right_val = evaluate_ast(binop.right, state, static_tools, custom_tools, authorized_imports)

    # Determine the operation based on the type of the operator in the BinOp
    if isinstance(binop.op, ast.Add):
        return left_val + right_val
    elif isinstance(binop.op, ast.Sub):
        return left_val - right_val
    elif isinstance(binop.op, ast.Mult):
        return left_val * right_val
    elif isinstance(binop.op, ast.Div):
        return left_val / right_val
    elif isinstance(binop.op, ast.Mod):
        return left_val % right_val
    elif isinstance(binop.op, ast.Pow):
        return left_val**right_val
    elif isinstance(binop.op, ast.FloorDiv):
        return left_val // right_val
    elif isinstance(binop.op, ast.BitAnd):
        return left_val & right_val
    elif isinstance(binop.op, ast.BitOr):
        return left_val | right_val
    elif isinstance(binop.op, ast.BitXor):
        return left_val ^ right_val
    elif isinstance(binop.op, ast.LShift):
        return left_val << right_val
    elif isinstance(binop.op, ast.RShift):
        return left_val >> right_val
    else:
        raise NotImplementedError(f"Binary operation {type(binop.op).__name__} is not implemented.")


def evaluate_assign(
    assign: ast.Assign,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    result = evaluate_ast(assign.value, state, static_tools, custom_tools, authorized_imports)
    if len(assign.targets) == 1:
        target = assign.targets[0]
        set_value(target, result, state, static_tools, custom_tools, authorized_imports)
    else:
        if len(assign.targets) != len(result):
            raise InterpreterError(f"Assign failed: expected {len(result)} values but got {len(assign.targets)}.")
        expanded_values = []
        for tgt in assign.targets:
            if isinstance(tgt, ast.Starred):
                expanded_values.extend(result)
            else:
                expanded_values.append(result)
        for tgt, val in zip(assign.targets, expanded_values):
            set_value(tgt, val, state, static_tools, custom_tools, authorized_imports)
    return result


def set_value(
    target: ast.AST,
    value: Any,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    if isinstance(target, ast.Name):
        if target.id in static_tools:
            raise InterpreterError(f"Cannot assign to name '{target.id}': doing this would erase the existing tool!")
        state[target.id] = value
    elif isinstance(target, ast.Tuple):
        if not isinstance(value, tuple):
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                value = tuple(value)
            else:
                raise InterpreterError("Cannot unpack non-tuple value")
        if len(target.elts) != len(value):
            raise InterpreterError("Cannot unpack tuple of wrong size")
        for i, elem in enumerate(target.elts):
            set_value(elem, value[i], state, static_tools, custom_tools, authorized_imports)
    elif isinstance(target, ast.Subscript):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
        obj[key] = value
    elif isinstance(target, ast.Attribute):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        setattr(obj, target.attr, value)


def evaluate_call(
    call: ast.Call,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    if not (
        isinstance(call.func, ast.Attribute) or isinstance(call.func, ast.Name) or isinstance(call.func, ast.Subscript)
    ):
        raise InterpreterError(f"This is not a correct function: {call.func}).")
    if isinstance(call.func, ast.Attribute):
        obj = evaluate_ast(call.func.value, state, static_tools, custom_tools, authorized_imports)
        func_name = call.func.attr
        if not hasattr(obj, func_name):
            raise InterpreterError(f"Object {obj} has no attribute {func_name}")
        func = getattr(obj, func_name)

    elif isinstance(call.func, ast.Name):
        func_name = call.func.id
        if func_name in state:
            func = state[func_name]
        elif func_name in static_tools:
            func = static_tools[func_name]
        elif func_name in custom_tools:
            func = custom_tools[func_name]
        elif func_name in ERRORS:
            func = ERRORS[func_name]
        else:
            raise InterpreterError(
                f"It is not permitted to evaluate {func_name} not present in the provided tools or functions defined/imported in previous code (tried to execute {call.func.id})."
            )

    elif isinstance(call.func, ast.Subscript):
        value = evaluate_ast(call.func.value, state, static_tools, custom_tools, authorized_imports)
        index = evaluate_ast(call.func.slice, state, static_tools, custom_tools, authorized_imports)
        if isinstance(value, (list, tuple)):
            func = value[index]
        else:
            raise InterpreterError(f"Cannot subscript object of type {type(value).__name__}")

        if not callable(func):
            raise InterpreterError(f"This is not a correct function: {call.func}).")
        func_name = None
    args = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            args.extend(evaluate_ast(arg.value, state, static_tools, custom_tools, authorized_imports))
        else:
            args.append(evaluate_ast(arg, state, static_tools, custom_tools, authorized_imports))

    kwargs = {
        keyword.arg: evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)
        for keyword in call.keywords
    }

    if func_name == "super":
        if not args:
            if "__class__" in state and "self" in state:
                return super(state["__class__"], state["self"])
            else:
                raise InterpreterError("super() needs at least one argument")
        cls = args[0]
        if not isinstance(cls, type):
            raise InterpreterError("super() argument 1 must be type")
        if len(args) == 1:
            return super(cls)
        elif len(args) == 2:
            instance = args[1]
            return super(cls, instance)
        else:
            raise InterpreterError("super() takes at most 2 arguments")
    else:
        if func_name == "print":
            output = " ".join(map(str, args))
            global PRINT_OUTPUTS
            PRINT_OUTPUTS += output + "\n"
            # cap the number of lines
            return None
        else:  # Assume it's a callable object
            if (
                (inspect.getmodule(func) == builtins)
                and inspect.isbuiltin(func)
                and (func not in static_tools.values())
            ):
                raise InterpreterError(
                    f"Invoking a builtin function that has not been explicitly added as a tool is not allowed ({func_name})."
                )
            return func(*args, **kwargs)


def evaluate_subscript(
    subscript: ast.Subscript,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    index = evaluate_ast(subscript.slice, state, static_tools, custom_tools, authorized_imports)
    value = evaluate_ast(subscript.value, state, static_tools, custom_tools, authorized_imports)

    if isinstance(value, str) and isinstance(index, str):
        raise InterpreterError("You're trying to subscript a string with a string index, which is impossible")
    if isinstance(value, pd.core.indexing._LocIndexer):
        parent_object = value.obj
        return parent_object.loc[index]
    if isinstance(value, pd.core.indexing._iLocIndexer):
        parent_object = value.obj
        return parent_object.iloc[index]
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return value[index]
    elif isinstance(value, pd.core.groupby.generic.DataFrameGroupBy):
        return value[index]
    elif isinstance(index, slice):
        return value[index]
    elif isinstance(value, (list, tuple)):
        if not (-len(value) <= index < len(value)):
            raise InterpreterError(f"Index {index} out of bounds for list of length {len(value)}")
        return value[int(index)]
    elif isinstance(value, str):
        if not (-len(value) <= index < len(value)):
            raise InterpreterError(f"Index {index} out of bounds for string of length {len(value)}")
        return value[index]
    else:
        error_message = f"Could not index {value} with '{index}'."
        if isinstance(index, str) and isinstance(value, Mapping):
            close_matches = difflib.get_close_matches(index, list(value.keys()))
            if len(close_matches) > 0:
                error_message += f" Maybe you meant one of these indexes instead: {str(close_matches)}"
        return value


def evaluate_name(
    name: ast.Name,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    if name.id in state:
        return state[name.id]
    elif name.id in static_tools:
        return static_tools[name.id]
    elif name.id in custom_tools:
        return custom_tools[name.id]
    elif name.id in ERRORS:
        return ERRORS[name.id]
    close_matches = difflib.get_close_matches(name.id, list(state.keys()))
    if len(close_matches) > 0:
        return state[close_matches[0]]
    raise InterpreterError(f"The variable `{name.id}` is not defined.")


def evaluate_condition(
    condition: ast.Compare,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> bool:
    left = evaluate_ast(condition.left, state, static_tools, custom_tools, authorized_imports)
    comparators = [
        evaluate_ast(c, state, static_tools, custom_tools, authorized_imports) for c in condition.comparators
    ]
    ops = [type(op) for op in condition.ops]

    result = True
    current_left = left

    for op, comparator in zip(ops, comparators):
        if op == ast.Eq:
            current_result = current_left == comparator
        elif op == ast.NotEq:
            current_result = current_left != comparator
        elif op == ast.Lt:
            current_result = current_left < comparator
        elif op == ast.LtE:
            current_result = current_left <= comparator
        elif op == ast.Gt:
            current_result = current_left > comparator
        elif op == ast.GtE:
            current_result = current_left >= comparator
        elif op == ast.Is:
            current_result = current_left is comparator
        elif op == ast.IsNot:
            current_result = current_left is not comparator
        elif op == ast.In:
            current_result = current_left in comparator
        elif op == ast.NotIn:
            current_result = current_left not in comparator
        else:
            raise InterpreterError(f"Operator not supported: {op}")

        result = result & current_result
        current_left = comparator

        if isinstance(result, bool) and not result:
            break

    return result if isinstance(result, (bool, pd.Series)) else result.all()


def evaluate_if(
    if_statement: ast.If,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    result = None
    test_result = evaluate_ast(if_statement.test, state, static_tools, custom_tools, authorized_imports)
    if test_result:
        for line in if_statement.body:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    else:
        for line in if_statement.orelse:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    return result


def evaluate_for(
    for_loop: ast.For,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    result = None
    iterator = evaluate_ast(for_loop.iter, state, static_tools, custom_tools, authorized_imports)
    for counter in iterator:
        set_value(
            for_loop.target,
            counter,
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
        for node in for_loop.body:
            try:
                line_result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
                if line_result is not None:
                    result = line_result
            except BreakException:
                break
            except ContinueException:
                continue
        else:
            continue
        break
    return result


def evaluate_listcomp(
    listcomp: ast.ListComp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> List[Any]:
    def inner_evaluate(generators: List[ast.comprehension], index: int, current_state: Dict[str, Any]) -> List[Any]:
        if index >= len(generators):
            return [
                evaluate_ast(
                    listcomp.elt,
                    current_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
            ]
        generator = generators[index]
        iter_value = evaluate_ast(
            generator.iter,
            current_state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
        result = []
        for value in iter_value:
            new_state = current_state.copy()
            if isinstance(generator.target, ast.Tuple):
                for idx, elem in enumerate(generator.target.elts):
                    new_state[elem.id] = value[idx]
            else:
                new_state[generator.target.id] = value
            if all(
                evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                for if_clause in generator.ifs
            ):
                result.extend(inner_evaluate(generators, index + 1, new_state))
        return result

    return inner_evaluate(listcomp.generators, 0, state)


def evaluate_try(
    try_node: ast.Try,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    try:
        for stmt in try_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        matched = False
        for handler in try_node.handlers:
            if handler.type is None or isinstance(
                e,
                evaluate_ast(handler.type, state, static_tools, custom_tools, authorized_imports),
            ):
                matched = True
                if handler.name:
                    state[handler.name] = e
                for stmt in handler.body:
                    evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
                break
        if not matched:
            raise e
    else:
        if try_node.orelse:
            for stmt in try_node.orelse:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    finally:
        if try_node.finalbody:
            for stmt in try_node.finalbody:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)


def evaluate_raise(
    raise_node: ast.Raise,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    if raise_node.exc is not None:
        exc = evaluate_ast(raise_node.exc, state, static_tools, custom_tools, authorized_imports)
    else:
        exc = None
    if raise_node.cause is not None:
        cause = evaluate_ast(raise_node.cause, state, static_tools, custom_tools, authorized_imports)
    else:
        cause = None
    if exc is not None:
        if cause is not None:
            raise exc from cause
        else:
            raise exc
    else:
        raise InterpreterError("Re-raise is not supported without an active exception")


def evaluate_assert(
    assert_node: ast.Assert,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    test_result = evaluate_ast(assert_node.test, state, static_tools, custom_tools, authorized_imports)
    if not test_result:
        if assert_node.msg:
            msg = evaluate_ast(assert_node.msg, state, static_tools, custom_tools, authorized_imports)
            raise AssertionError(msg)
        else:
            # Include the failing condition in the assertion message
            test_code = ast.unparse(assert_node.test)
            raise AssertionError(f"Assertion failed: {test_code}")


def evaluate_with(
    with_node: ast.With,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    contexts = []
    for item in with_node.items:
        context_expr = evaluate_ast(item.context_expr, state, static_tools, custom_tools, authorized_imports)
        if item.optional_vars:
            state[item.optional_vars.id] = context_expr.__enter__()
            contexts.append(state[item.optional_vars.id])
        else:
            context_var = context_expr.__enter__()
            contexts.append(context_var)

    try:
        for stmt in with_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        for context in reversed(contexts):
            context.__exit__(type(e), e, e.__traceback__)
        raise
    else:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


def get_safe_module(raw_module, dangerous_patterns, authorized_imports, visited=None, depth=0):
    """Creates a safe copy of a module with depth limiting and error handling"""
    # Max depth to prevent infinite recursion with packages like TF/Keras
    MAX_DEPTH = 3
    
    if not isinstance(raw_module, ModuleType) or depth > MAX_DEPTH:
        return raw_module

    if visited is None:
        visited = set()

    module_id = id(raw_module)
    if module_id in visited:
        return raw_module
    visited.add(module_id)

    try:
        safe_module = ModuleType(raw_module.__name__)
    except AttributeError:
        return raw_module

    # Get attributes without triggering full module load
    try:
        attr_names = dir(raw_module)
    except Exception as e:
        logger.warning(f"Couldn't list attributes for {raw_module}: {str(e)}")
        return safe_module

    for attr_name in attr_names:
        # Skip dangerous attributes
        if any(p in attr_name for p in dangerous_patterns):
            continue

        try:
            attr_value = getattr(raw_module, attr_name)
        except Exception as e:
            logger.debug(f"Skipping attribute {attr_name} in {raw_module}: {str(e)}")
            continue

        # Handle submodules with increased depth counter
        if isinstance(attr_value, ModuleType):
            attr_value = get_safe_module(
                attr_value, 
                dangerous_patterns,
                authorized_imports,
                visited=visited,
                depth=depth + 1
            )

        setattr(safe_module, attr_name, attr_value)

    return safe_module


def import_modules(expression, state, authorized_imports):
    dangerous_patterns = (
        "_os",
        "os",
        "subprocess",
        "_subprocess",
        "pty",
        "system",
        "popen",
        "spawn",
        "shutil",
        "sys",
        "pathlib",
        "io",
        "socket",
        "compile",
        "eval",
        "exec",
        "multiprocessing",
        "__internal__"
    )

    def check_module_authorized(module_name):
        if "*" in authorized_imports:
            return True
        else:
            module_path = module_name.split(".")
            if any([module in dangerous_patterns and module not in authorized_imports for module in module_path]):
                return False
            module_subpaths = [".".join(module_path[:i]) for i in range(1, len(module_path) + 1)]
            return any(subpath in authorized_imports for subpath in module_subpaths)

    if isinstance(expression, ast.Import):
        for alias in expression.names:
            if check_module_authorized(alias.name):
                raw_module = import_module(alias.name)
                state[alias.asname or alias.name] = get_safe_module(raw_module, dangerous_patterns, authorized_imports)
            else:
                raise InterpreterError(
                    f"Import of {alias.name} is not allowed. Authorized imports are: {str(authorized_imports)}"
                )
        return None
    elif isinstance(expression, ast.ImportFrom):
        if check_module_authorized(expression.module):
            raw_module = __import__(expression.module, fromlist=[alias.name for alias in expression.names])
            module = get_safe_module(raw_module, dangerous_patterns, authorized_imports)
            if expression.names[0].name == "*":  # Handle "from module import *"
                if hasattr(module, "__all__"):  # If module has __all__, import only those names
                    for name in module.__all__:
                        state[name] = getattr(module, name)
                else:  # If no __all__, import all public names (those not starting with '_')
                    for name in dir(module):
                        if not name.startswith("_"):
                            state[name] = getattr(module, name)
            else:  # regular from imports
                for alias in expression.names:
                    if hasattr(module, alias.name):
                        state[alias.asname or alias.name] = getattr(module, alias.name)
                    else:
                        raise InterpreterError(f"Module {expression.module} has no attribute {alias.name}")
        else:
            raise InterpreterError(f"Import from {expression.module} is not allowed.")
        return None


def evaluate_dictcomp(
    dictcomp: ast.DictComp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Dict[Any, Any]:
    result = {}
    for gen in dictcomp.generators:
        iter_value = evaluate_ast(gen.iter, state, static_tools, custom_tools, authorized_imports)
        for value in iter_value:
            new_state = state.copy()
            set_value(
                gen.target,
                value,
                new_state,
                static_tools,
                custom_tools,
                authorized_imports,
            )
            if all(
                evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                for if_clause in gen.ifs
            ):
                key = evaluate_ast(
                    dictcomp.key,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                val = evaluate_ast(
                    dictcomp.value,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                result[key] = val
    return result


def evaluate_ast(
    expression: ast.AST,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str] = BASE_BUILTIN_MODULES,
):
    """
    Evaluate an abstract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abstract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignments.
        static_tools (`Dict[str, Callable]`):
            Functions that may be called during the evaluation. Trying to change one of these static_tools will raise an error.
        custom_tools (`Dict[str, Callable]`):
            Functions that may be called during the evaluation. These static_tools can be overwritten.
        authorized_imports (`List[str]`):
            The list of modules that can be imported by the code. By default, only a few safe modules are allowed.
            If it contains "*", it will authorize any import. Use this at your own risk!
    """
    global OPERATIONS_COUNT
    if OPERATIONS_COUNT >= MAX_OPERATIONS:
        raise InterpreterError(
            f"Reached the max number of operations of {MAX_OPERATIONS}. Maybe there is an infinite loop somewhere in the code, or you're just asking too many calculations."
        )
    OPERATIONS_COUNT += 1
    if isinstance(expression, ast.Assign):
        # Assignment -> we evaluate the assignment which should update the state
        # We return the variable assigned as it may be used to determine the final result.
        return evaluate_assign(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.AugAssign):
        return evaluate_augassign(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Call):
        # Function call -> we return the value of the function call
        return evaluate_call(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Constant):
        # Constant -> just return the value
        return expression.value
    elif isinstance(expression, ast.Tuple):
        return tuple(
            evaluate_ast(elt, state, static_tools, custom_tools, authorized_imports) for elt in expression.elts
        )
    elif isinstance(expression, (ast.ListComp, ast.GeneratorExp)):
        return evaluate_listcomp(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.UnaryOp):
        return evaluate_unaryop(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Starred):
        return evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.BoolOp):
        # Boolean operation -> evaluate the operation
        return evaluate_boolop(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Break):
        raise BreakException()
    elif isinstance(expression, ast.Continue):
        raise ContinueException()
    elif isinstance(expression, ast.BinOp):
        # Binary operation -> execute operation
        return evaluate_binop(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Compare):
        # Comparison -> evaluate the comparison
        return evaluate_condition(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Lambda):
        return evaluate_lambda(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.FunctionDef):
        return evaluate_function_def(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Dict):
        # Dict -> evaluate all keys and values
        keys = [evaluate_ast(k, state, static_tools, custom_tools, authorized_imports) for k in expression.keys]
        values = [evaluate_ast(v, state, static_tools, custom_tools, authorized_imports) for v in expression.values]
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        # Expression -> evaluate the content
        return evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.For):
        # For loop -> execute the loop
        return evaluate_for(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.FormattedValue):
        # Formatted value (part of f-string) -> evaluate the content and return
        return evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.If):
        # If -> execute the right branch
        return evaluate_if(expression, state, static_tools, custom_tools, authorized_imports)
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        return evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.JoinedStr):
        return "".join(
            [str(evaluate_ast(v, state, static_tools, custom_tools, authorized_imports)) for v in expression.values]
        )
    elif isinstance(expression, ast.List):
        # List -> evaluate all elements
        return [evaluate_ast(elt, state, static_tools, custom_tools, authorized_imports) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        # Name -> pick up the value in the state
        return evaluate_name(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Subscript):
        # Subscript -> return the value of the indexing
        return evaluate_subscript(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.IfExp):
        test_val = evaluate_ast(expression.test, state, static_tools, custom_tools, authorized_imports)
        if test_val:
            return evaluate_ast(expression.body, state, static_tools, custom_tools, authorized_imports)
        else:
            return evaluate_ast(expression.orelse, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Attribute):
        value = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
        return getattr(value, expression.attr)
    elif isinstance(expression, ast.Slice):
        return slice(
            evaluate_ast(expression.lower, state, static_tools, custom_tools, authorized_imports)
            if expression.lower is not None
            else None,
            evaluate_ast(expression.upper, state, static_tools, custom_tools, authorized_imports)
            if expression.upper is not None
            else None,
            evaluate_ast(expression.step, state, static_tools, custom_tools, authorized_imports)
            if expression.step is not None
            else None,
        )
    elif isinstance(expression, ast.DictComp):
        return evaluate_dictcomp(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.While):
        return evaluate_while(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, (ast.Import, ast.ImportFrom)):
        return import_modules(expression, state, authorized_imports)
    elif isinstance(expression, ast.ClassDef):
        return evaluate_class_def(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Try):
        return evaluate_try(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Raise):
        return evaluate_raise(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Assert):
        return evaluate_assert(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.With):
        return evaluate_with(expression, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(expression, ast.Set):
        return {evaluate_ast(elt, state, static_tools, custom_tools, authorized_imports) for elt in expression.elts}
    elif isinstance(expression, ast.Return):
        raise ReturnException(
            evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
            if expression.value
            else None
        )
    elif isinstance(expression, ast.Pass):
        return None
    else:
        # For now we refuse anything else. Let's add things as we need them.
        raise InterpreterError(f"{expression.__class__.__name__} is not supported.")


def evaluate_python_code(
    code: str,
    static_tools: Optional[Dict[str, Callable]] = None,
    custom_tools: Optional[Dict[str, Callable]] = None,
    state: Optional[Dict[str, Any]] = None,
    authorized_imports: List[str] = BASE_BUILTIN_MODULES,
    max_print_outputs_length: int = 50000,
) -> Any:
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (str): The code to evaluate.
        static_tools (Optional[Dict[str, Callable]]): The functions that may be called during the evaluation.
        custom_tools (Optional[Dict[str, Callable]]): The functions that may be called during the evaluation.
        state (Optional[Dict[str, Any]]): A dictionary mapping variable names to values.
        authorized_imports (List[str]): List of modules that are allowed to be imported.
        max_print_outputs_length (int): Maximum length for the captured print outputs.
    """
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Code execution failed on line {e.lineno} due to: {type(e).__name__}\n"
            f"{e.text}"
            f"{' ' * (e.offset or 0)}^\n"
            f"Error: {str(e)}"
        )

    if state is None:
        state = {}
    static_tools = static_tools.copy() if static_tools is not None else {}
    custom_tools = custom_tools if custom_tools is not None else {}
    result = None
    global PRINT_OUTPUTS
    PRINT_OUTPUTS = ""
    global OPERATIONS_COUNT
    OPERATIONS_COUNT = 0

    try:
        for node in expression.body:
            result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
        state["print_outputs"] = truncate_content(PRINT_OUTPUTS, max_length=max_print_outputs_length)
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        error_content = truncate_content(PRINT_OUTPUTS, max_length=max_print_outputs_length)
        error_msg = (
            f"Code execution failed at node '{ast.get_source_segment(code, node)}' due to exception:\n{error_trace}",
            PRINT_OUTPUTS,
            error_content
        )
        raise InterpreterError(error_msg)
    
    
def fix_final_answer_code(code: str) -> str:
    """
    Sometimes an LLM can try to assign a variable to final_answer, which would break the final_answer() tool.
    This function fixes this behaviour by replacing variable assignments to final_answer with final_answer_variable,
    while preserving function calls to final_answer().
    """
    # First, find if there's a direct assignment to final_answer
    # Use word boundary and negative lookbehind to ensure it's not an object attribute
    assignment_pattern = r"(?<!\.)(?<!\w)\bfinal_answer\s*="
    if "final_answer(" not in code or not re.search(assignment_pattern, code):
        # If final_answer tool is not called in this blob, then doing the replacement is hazardous because it could false the model's memory for next steps.
        # Let's not modify the code and leave the subsequent assignment error happen.
        return code

    # Pattern for replacing variable assignments
    # Looks for 'final_answer' followed by '=' with optional whitespace
    # Negative lookbehind ensures we don't match object attributes
    assignment_regex = r"(?<!\.)(?<!\w)(\bfinal_answer)(\s*=)"
    code = re.sub(assignment_regex, r"final_answer_variable\2", code)

    # Pattern for replacing variable usage but not function calls
    # Negative lookahead (?!\s*\() ensures we don't match function calls
    # Negative lookbehind (?<!\.|\w) ensures we don't match object methods or other variables
    variable_regex = r"(?<!\.)(?<!\w)(\bfinal_answer\b)(?!\s*\()"
    code = re.sub(variable_regex, "final_answer_variable", code)
    return code
    
class LocalPythonInterpreter(ExecutionEnvironment):
    
    language: str = "python"
    
    def __init__(
        self,
        session_id: str,
        mcp_servers: Dict,
        additional_authorized_imports: List[str],
    ):
        self.custom_tools = {}
        self.state = {}
        self.additional_authorized_imports = additional_authorized_imports
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports) | {"tensorflow", "keras"})
        # Add base trusted tools to list
        self.static_tools = {
            **mcp_servers,
            **BASE_PYTHON_TOOLS.copy(),
        }
        # IMPROVE: assert self.authorized imports are all installed locally
        super().__init__(session_id, mcp_servers)
        
    def __call__(self, code_action: str, additional_variables: Dict, return_type: str = "string") -> Tuple[Union[List[dict], str], str, bool]:
        self.state.update(additional_variables)
        output = evaluate_python_code(
            code_action,
            static_tools=self.static_tools,
            custom_tools=self.custom_tools,
            state=self.state,
            authorized_imports=self.authorized_imports,
        )
        logs = self.state["print_outputs"]
        if return_type == "string" and output is not None:
            if isinstance(output, pd.DataFrame):
                markdown_table = output.head(10).to_markdown()
            elif isinstance(output, pd.Series):
                markdown_table = output.to_frame().head(10).T.to_markdown()
            elif isinstance(output, (list, tuple)):
                markdown_table = pd.DataFrame(output).head(10).to_markdown()
            else:
                markdown_table = json_to_markdown(json.dumps(output))
            logger.error(f"Python Environment Markdown Table: {markdown_table}")
            return markdown_table, logs, False
        else:
            return output, logs, False

    def attach_variables(self, variables: dict):
        self.state.update(variables)

    def attach_mcp_servers(self, mcp_servers: Dict[str, ModelContextProtocolImpl]):
        self.static_tools.update(mcp_servers)

    def parse_error_logs(self, execution_logs: str) -> str:
        # Regex pattern to capture the full InterpreterError including multiline messages
        pattern = r'InterpreterError:\s*(.*?)\n(?:LINE|\^)'
        match = re.search(pattern, execution_logs, re.DOTALL)
        if match:
            # Replace excessive whitespace/newlines with a single space for readability
            error_msg = ' '.join(match.group(1).split())
            return error_msg
        else:
            return execution_logs

    
    def _extract_markdown_code(self, code_blob: str) -> str:
        """Extracts Python code from markdown code block"""
        match = re.search(r"```python\s*\n(.+?)\n*```", code_blob, re.DOTALL)
        if not match:
            raise ValueError("No valid Python code or markdown code block found")
        return match.group(1)

    def _collect_imports(self, tree: ast.AST) -> Set[str]:
        """Collects root package names from all import statements"""
        packages = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    packages.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                packages.add(node.module.split('.')[0])
        return packages

    def _install_missing_modules(self, modules: Set[str]):
        """Installs packages that aren't available in the environment"""
        for module in modules:
            if not self._is_module_installed(module):
                logger.info(f"Installing missing module: {module}")
                pip_package_name = get_pip_package(module)
                logger.info(f"Installing pip package: {pip_package_name}")
                if not self._pip_install(pip_package_name):
                    raise ImportError(f"Failed to install required module: {module}")

    def _is_module_installed(self, package: str) -> bool:
        """Checks if a package is importable"""
        try:
            import_module(package)
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"Error checking package {package}: {e}")
            return False

    def _pip_install(self, package: str) -> bool:
        """Executes pip install command"""
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def parse_function(self, code_blob: str) -> Dict[str, Callable]:
        """
        Parses the given code blob using Python's ast module and returns a dictionary mapping
        function names to their corresponding callable functions.
        
        If the input is not valid Python code, it first attempts to extract the substring inside
        a markdown code block (```python ... ```).

        Parameters:
            code_blob (str): A string containing Python code or markdown with a python code block.

        Returns:
            Dict[str, Callable]: A dictionary where keys are function names and values are callable functions.
        """
        try:
            # Attempt to parse the code blob directly.
            tree = ast.parse(code_blob)
            code_to_compile = code_blob
        except SyntaxError:
            # If parsing fails, extract the code block from markdown.
            code_to_compile = self._extract_markdown_code(code_blob)
            try:
                tree = ast.parse(code_to_compile)
            except SyntaxError as e:
                raise ValueError(f"Extracted code is not valid Python: {e}")
            
        # Identify and install required modules
        required_modules = self._collect_imports(tree)
        self._install_missing_modules(required_modules)
        # Compile and execute the (valid) code in a temporary namespace.
        namespace = {}
        compiled_code = compile(tree, filename="<ast>", mode="exec")
        exec(compiled_code, namespace)

        # Filter the namespace for functions (ignoring imported modules and other objects).
        functions_dict = {
            name: obj
            for name, obj in namespace.items()
            if callable(obj) and isinstance(obj, type(lambda: None))
        }
        return functions_dict
        
    def parse_code_blobs(self, code_blob: str) -> str:
        """Parses the LLM's output to get any code blob inside. Will return the code directly if it's code."""
        pattern = r"```(?:python)?\n(.*?)\n```"
        matches = re.findall(pattern, code_blob, re.DOTALL)
        if len(matches) == 0:
            try:  # Maybe the LLM outputted a code blob directly
                ast.parse(code_blob)
                return code_blob
            except SyntaxError:
                pass
            if "final" in code_blob and "answer" in code_blob:
                raise ValueError(f"""
                        Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
                        Here is your code snippet:
                        {code_blob}
                        It seems like you're trying to return the final answer, you can do it as follows:
                        Code:
                        ```py
                        final_answer("YOUR FINAL ANSWER HERE")
                        ```<end_code>""".strip())
            raise ValueError(f"""Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
                    Here is your code snippet:
                    {code_blob}
                    Make sure to include code with the correct pattern, for instance:
                    Thoughts: Your thoughts
                    Code:
                    ```py
                    # Your python code here
                    ```<end_code>""".strip())
        return fix_final_answer_code("\n\n".join(match.strip() for match in matches))

    def get_storage_id(self, next_step_id: int) -> str:
        return f"temp_storage_{next_step_id}"
    
    def set_storage(self, next_step_id: int, code_action: str):
        variable_name = self.get_storage_id(next_step_id)
        try:
            # Execute the code_action and capture the result
            result = evaluate_python_code(
                code_action.strip(),
                static_tools=self.static_tools,
                custom_tools=self.custom_tools,
                state=self.state,
                authorized_imports=self.authorized_imports,
                max_print_outputs_length=self.max_print_outputs_length,
            )
            # Write to environment state
            self.state[variable_name] = result
            # Write to storage
            self.storage[next_step_id] = variable_name
        except InterpreterError as e:
            self.state[variable_name] = None
            raise ValueError(f"Error storing data in {variable_name}: {str(e)}")
        
    def reset_storage(self):
        self.storage = {}
    
    def get_final_storage(self) -> pd.DataFrame:
        max_step_id = max(self.storage.keys())
        variable_name = self.get_storage_id(max_step_id)
        result = self.state.get(variable_name, None)
        if result is None:
            pd.DataFrame()
        if isinstance(result, pd.DataFrame):
            return result
        if isinstance(result, pd.Series):
            return result.to_frame().T
        if isinstance(result, (list, tuple)):
            return pd.DataFrame(result)
        if isinstance(result, dict):
            return pd.DataFrame(result)
        raise ValueError(f"Final storage is not a DataFrame, but a {type(result)}")
    
    def get_storage(self, next_step_id: int) -> str:
        variable_name = self.get_storage_id(next_step_id)
        variable_value = truncate_content(str(self.state.get(variable_name, None)), max_length=self.max_print_outputs_length)
        trimmed = False if len(variable_value) < len(str(self.state.get(variable_name, None))) else True
        return f"""
        Variable to access result of step {next_step_id}:
            {variable_name} = {variable_value}{... if trimmed else ""}
        """
        
__all__ = ["evaluate_python_code", "LocalPythonInterpreter"]
