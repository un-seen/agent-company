import builtins

class InterpreterError(ValueError):
    """
    An error raised when the interpreter cannot evaluate a SQL expression, due to syntax error or unsupported
    operations.
    """

    pass

class BreakException(Exception):
    pass

class ContinueException(Exception):
    pass

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value
        
ERRORS = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type) and issubclass(getattr(builtins, name), BaseException)
}