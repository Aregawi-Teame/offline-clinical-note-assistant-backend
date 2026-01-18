"""
Logging configuration with request_id support.
"""
import logging
import sys
from typing import Optional


class RequestIDFormatter(logging.Formatter):
    """Custom formatter that includes request_id when available."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record, including request_id if present."""
        # Format using the base formatter
        formatted = super().format(record)
        
        # If request_id exists, prepend it to the formatted message
        if hasattr(record, "request_id") and record.request_id:
            # Insert request_id after timestamp but before message
            # Format: timestamp - name - level - [request_id] - message
            parts = formatted.split(" - ", 3)
            if len(parts) >= 4:
                formatted = f"{parts[0]} - {parts[1]} - {parts[2]} - [{record.request_id}] - {parts[3]}"
            else:
                # Fallback: prepend request_id
                formatted = f"[{record.request_id}] {formatted}"
        
        return formatted


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Setup application logging configuration.
    
    Args:
        log_level: Logging level (defaults to INFO)
        log_format: Log format string (defaults to readable format)
    """
    # Set default log level to INFO
    level = log_level or "INFO"
    
    # Get log level from string
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create console handler with readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Use custom formatter that supports request_id
    if log_format:
        formatter = logging.Formatter(log_format)
    else:
        # Default readable format with request_id support
        formatter = RequestIDFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Prevent duplicate logs from uvicorn
    logging.getLogger("uvicorn.access").propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
