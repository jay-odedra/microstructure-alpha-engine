from loguru import logger
import sys
from pathlib import Path


def setup_logger(log_file: str | Path = "app.log"):

    logger.remove()

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
    )

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="10 MB",
    )

    return logger
