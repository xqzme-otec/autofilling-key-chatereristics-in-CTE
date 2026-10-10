"""Entrypoint: python -m ml.service (listens on HOST:PORT, default 0.0.0.0:8001)."""
import logging
import sys

import uvicorn

from ml.service.app import create_app
from ml.service.config import ConfigError, Settings
from ml.service.logs import setup_logging


def main() -> int:
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        setup_logging("INFO")
        logging.getLogger("ml.service").error("invalid configuration: %s", e)
        return 2
    app = create_app(settings)
    # log_config=None keeps our root handler and format for uvicorn's own loggers;
    # access lines are written by RequestContextMiddleware (with request_id and duration)
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None, access_log=False,
                log_level=settings.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())
