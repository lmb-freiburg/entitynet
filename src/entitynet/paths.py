"""
Loads environment variables for entitynet project.

Define ENTITYNET_DATA_DIR with the base directory for all datasets.
Optionally define ENTITYNET_OUTPUT_DIR for experiment output.
Optionally define ENTITYNET_CACHE_DIR to put cache in a different location.
Git repository root is inferred from the location of this file.
"""

import os
from pathlib import Path

_repo_root: Path = Path(__file__).parent.parent.parent.resolve().absolute().as_posix()


def get_entitynet_repo_root():
    return Path(_repo_root)


def get_entitynet_annotations_dir():
    return get_entitynet_repo_root() / "src/entitynet/annotations"


def get_entitynet_data_dir() -> Path:
    data_dir = os.environ.get("ENTITYNET_DATA_DIR")
    if data_dir is None:
        raise ValueError(
            "ENTITYNET_DATA_DIR environment variable not set. Set it, or overwrite function "
            "get_entitynet_data_dir()"
        )
    return Path(data_dir)


def get_entitynet_output_dir():
    output_dir = os.environ.get("ENTITYNET_OUTPUT_DIR", get_entitynet_data_dir() / "output")
    if output_dir is None:
        raise ValueError(
            "ENTITYNET_OUTPUT_DIR environment variable not set. Set it, or overwrite function "
            "get_entitynet_output_dir()"
        )
    return Path(output_dir)


def get_entitynet_cache_dir() -> Path:
    cache_dir = os.environ.get("ENTITYNET_CACHE_DIR", get_entitynet_data_dir() / "cache")
    if cache_dir is None:
        raise ValueError(
            "ENTITYNET_CACHE_DIR environment variable not set. Set it, or overwrite function "
            "get_entitynet_cache_dir()"
        )
    return Path(cache_dir)
