"""
Utilities for neptune.ai logger
"""

import os

from packg.log import logger


def rebuild_dict_no_nones(input_dict):
    """
    Neptune logger can crash if an input dict contains None values. This function skips
    key-value pairs where the value is None.
    """
    if isinstance(input_dict, dict):
        new_dict = {}
        for k, v in input_dict.items():
            if v is None:
                continue
            new_dict[k] = rebuild_dict_no_nones(v)
        return new_dict
    return input_dict


def get_neptune_project():
    project = os.environ.get("NEPTUNE_PROJECT")
    if project is None:
        project = "gings/default"
        logger.info(f"NEPTUNE_PROJECT env variable is not set, using default: {project}")
    else:
        logger.info(f"Using neptune project from env variable NEPTUNE_PROJECT: {project}")
    return project
