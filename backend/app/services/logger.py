import logging
import sys

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for a service module.

    Usage:
        from app.services.logger import get_logger
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _FlushHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
