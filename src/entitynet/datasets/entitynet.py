from attr import define
from entitynet.paths import get_entitynet_data_dir
from datasets import load_dataset
from packg.system.systemcall import systemcall_with_assert
from typedparser import VerboseQuietArgs, add_argument

from attrs import define
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import VerboseQuietArgs, add_argument
from loguru import logger


LEGAL_SPLITS = {"train", "val", "test", "minitrain"}
ENTITYNET_HF_URL = "https://huggingface.co/datasets/lmb-freiburg/entitynet"
MAX_SIDE = 512
N_SHARDS_V10 = {
    "train": 2048,
    "val": 32,
    "test": 32,
    "minitrain": 64,
}

@define
class EntityNetUrlBuildArgs(VerboseQuietArgs):
    splits: str = add_argument(
        default="val",
        help="Which splits to download, comma-separated for multiple.",
    )
    workers: int = add_argument(
        shortcut="-w",
        type=int,
        default=8,
        help="Number of worker processes for downloading images.",
    )


def parse_args_for_url_build(args: EntityNetUrlBuildArgs):
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    entitynet_dir = ensure_entitynet_huggingface()
    split_list = [s.strip() for s in args.splits.split(",")]
    for s in split_list:
        if s not in LEGAL_SPLITS:
            raise ValueError(f"Invalid split: {s}. Must be one of {', '.join(LEGAL_SPLITS)}.")
    return entitynet_dir, split_list

def ensure_entitynet_huggingface():
    """
    Ensures the entitynet dataset is downloaded from HuggingFace into the correct data directory.
    """
    data_dir = get_entitynet_data_dir()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"EntityNet data dir does not exist: {data_dir}")
    entitynet_dir = data_dir / "entitynet"

    manual_setup = f"cd {data_dir}\ngit clone {ENTITYNET_HF_URL}"
    err_msg = (
        f"Folder {entitynet_dir} is not setup correctly, delete and re-run, or make sure it "
        f"contains the content from the huggingface repository. Manual setup:\n{manual_setup}"
    )
    if entitynet_dir.exists():
        assert (
            entitynet_dir / "images_metadata" / "train-00031-of-00032.parquet"
        ).is_file(), f"{err_msg}"
        assert (entitynet_dir / "entitynet-queries.parquet").is_file(), f"{err_msg}"
        return entitynet_dir

    try:
        systemcall_with_assert("git --version")
        systemcall_with_assert("git lfs version")
        logger.info(f"Downloading {ENTITYNET_HF_URL} to {entitynet_dir}, might take a while...")
        systemcall_with_assert(f"git clone {ENTITYNET_HF_URL} {entitynet_dir}")
    except AssertionError as e:
        raise RuntimeError(
            f"Git or Git LFS not installed or git clone failed. Install git and git lfs then "
            f"run again, or clone it yourself:\n{manual_setup}"
        )

    return entitynet_dir
