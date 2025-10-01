"""
Build entitynet from URLS, step 1: download EntityNet metadata from huggingface
"""

from loguru import logger
from entitynet.datasets.entitynet import EntityNetUrlBuildArgs, parse_args_for_url_build
from typedparser import TypedParser

N_RETRIES = 3
SLEEP_SEC = 1


def main():
    parser = TypedParser.create_parser(EntityNetUrlBuildArgs, description=__doc__)
    args: EntityNetUrlBuildArgs = parser.parse_args()
    logger.info(f"{args}")
    entitynet_dir, split_list = parse_args_for_url_build(args)
    logger.info(f"Done setting up dataset from huggingface into {entitynet_dir}")


if __name__ == "__main__":
    main()
