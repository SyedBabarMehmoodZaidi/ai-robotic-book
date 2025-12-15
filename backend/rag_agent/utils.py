import uuid
import time
from typing import Callable, Any
from functools import wraps


def generate_request_id() -> str:
    """
    Generate a unique request ID.
    """
    return str(uuid.uuid4())


def get_current_timestamp() -> str:
    """
    Get the current timestamp in ISO format.
    """
    from datetime import datetime
    return datetime.utcnow().isoformat()


def time_it(func: Callable) -> Callable:
    """
    Decorator to time a function's execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


def measure_time() -> float:
    """
    Get the current time for measuring execution time.
    """
    return time.time()


def format_elapsed_time(start_time: float) -> float:
    """
    Calculate the elapsed time from a start time.
    """
    return time.time() - start_time


class Timer:
    """
    A simple timer class for measuring execution time.
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()

    def elapsed(self) -> float:
        """Get the elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time