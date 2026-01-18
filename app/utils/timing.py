"""
Utility for timing operations.
"""
import time
from typing import Optional


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        """Initialize timer."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing."""
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    @property
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        return self.elapsed * 1000.0
