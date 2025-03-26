import math

from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

@tool
def sin(a: float) -> float:
    """Take the sine of a number."""
    return math.sin(a)

@tool
def cos(a: float) -> float:
    """Take the cosine of a number."""
    return math.cos(a)

@tool
def radians(a: float) -> float:
    """Convert degrees to radians."""
    return math.radians(a)

@tool
def exponentiation(a: float, b: float) -> float:
    """Raise one number to the power of another."""
    return a**b

@tool
def sqrt(a: float) -> float:
    """Take the square root of a number."""
    return math.sqrt(a)

@tool
def ceil(a: float) -> float:
    """Round a number up to the nearest integer."""
    return math.ceil(a)

@tool
def floor(a: float) -> float:
    """Round a number down to the nearest integer."""
    return math.floor(a)

@tool
def round(a: float) -> float:
    """Round a number to the nearest integer."""
    return math.round(a)

@tool
def log(a: float) -> float:
    """Take the natural logarithm of a number."""
    return math.log(a)
