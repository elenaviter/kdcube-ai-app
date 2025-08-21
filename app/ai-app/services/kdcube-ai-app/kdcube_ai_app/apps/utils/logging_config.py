# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

import logging
import os


def configure_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = os.getenv('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s')

    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    level = level_map.get(log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )


configure_logging()