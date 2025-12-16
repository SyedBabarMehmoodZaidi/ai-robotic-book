"""
Logging configuration for the application.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logging():
    """Set up logging configuration for the application."""
    # Create a custom logger
    logger = logging.getLogger('rag_qdrant_pipeline')
    logger.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    # Create console handler with higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Create file handler which logs even debug messages
    file_handler = RotatingFileHandler(
        'rag_pipeline.log',
        maxBytes=1024*1024*5,  # 5MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Initialize the logger
logger = setup_logging()