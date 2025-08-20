import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from lightning_utilities.core.rank_zero import log as lit_cmd_logger

from packg.dtime import get_timestamp_for_filename
from packg.log import SHORTER_FORMAT, SHORTEST_FORMAT, configure_logger
from visiontext.distutils import is_main_process


def setup_loguru_train_logging(output_dir, logger_level, log_name="log"):
    output_dir = Path(output_dir)
    if not is_main_process():
        # logger.info(f"Disabling logger for rank {get_rank()}")
        logger_level = logging.ERROR
        configure_logger(level=logger_level)
    else:
        # connect loguru.logger and lightning_utilities.core.rank_zero.log to file, only for rank 0
        setup_logger(output_dir, level=logger_level, log_name=log_name)


def setup_logger(output_dir, level="INFO", log_name="log"):
    datetime_str = get_timestamp_for_filename()
    log_file = output_dir / "logs" / f"{log_name}-{datetime_str}.log"
    os.makedirs(log_file.parent, exist_ok=True)
    print(f"Start logging to file:{log_file}")
    configure_logger(
        level=level,
        format=SHORTEST_FORMAT,
        add_sinks=[{"sink": log_file, "format": SHORTER_FORMAT, "colorize": False, "level": level}],
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-4s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    lit_cmd_logger.addHandler(file_handler)
    return log_file


def setup_seeds(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # torch.use_deterministic_algorithms(False)


def figure_out_world_size(config):
    """
    figure out the correct world size.
    note that at this point the worldsize is always 1, because the distributed init is not done yet.
    however certain datasets and losses need to know the future world_size so we set it here.

    """
    if config.trainer.devices == "auto":
        raise ValueError(
            f"{config.trainer.devices=} but it must be an integer or list to infer world size."
        )
    elif isinstance(config.trainer.devices, list):
        n_devices = len(config.trainer.devices)
    else:
        n_devices = int(config.trainer.devices)

    if not isinstance(config.trainer.num_nodes, int):
        raise ValueError(
            f"{config.trainer.num_nodes=} but it must be an integer to infer world size."
        )
    new_world_size = n_devices * config.trainer.num_nodes
    return new_world_size
