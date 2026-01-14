class THLError(Exception):
    """Base exception for all THL errors."""
    def __init__(self, message: str, hint: str = None):
        if hint:
            message = f"{message}\n\n💡 Hint: {hint}"
        super().__init__(message)

class THLConfigError(THLError):
    """Raised when configuration parameters are invalid."""
    pass

class THLMemoryError(THLError):
    """Raised when memory operations fail (e.g., out of bounds, invalid slots)."""
    pass

class THLRuntimeError(THLError):
    """Raised during model execution for runtime issues."""
    pass
