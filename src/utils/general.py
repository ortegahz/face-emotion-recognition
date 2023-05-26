# -*- coding: utf-8 -*-
import logging
import os
import sys


def set_logging(rank=0, dir_root='run', name='log', b_file=False):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter(f"{name}: %(asctime)s-%(message)s")
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_stream)
        if b_file:
            os.makedirs(dir_root, exist_ok=True)
            handler_file = logging.FileHandler(os.path.join(dir_root, f"{name}.log"))
            handler_file.setFormatter(formatter)
            log_root.addHandler(handler_file)

