# -*- coding: utf-8 -*-
import logging


def set_logging(name=None):
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    return logging.getLogger(name)


LOGGER = set_logging()
