# config.py
import logging
import sys
from pathlib import Path

# Directories
ROOT_DIR = "."
LOGS_DIR = Path(ROOT_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "fdd029d6a560451d80836002e7465914"

# Config MLflow
# MODEL_REGISTRY = Path(f"{EFS_DIR}/mlflow")
# Path(MODEL_REGISTRY).mkdir(parents=True, exist_ok=True)
# MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            # pylint: disable=line-too-long
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

# Logger
logging.config.dictConfig(logging_config)
logger = logging.getLogger()

# Constraints
