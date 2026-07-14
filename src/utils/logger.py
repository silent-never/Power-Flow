"""Simple logger scaffold."""

import logging

logger = logging.getLogger("powerflow")

def setup(level=logging.INFO):
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
