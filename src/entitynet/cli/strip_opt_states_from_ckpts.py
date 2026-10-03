"""
Create a copy of checkpoints with only model weights (no optimizer states)


"""

from pathlib import Path
from typing import Optional

from attrs import define
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from entitynet.litext.ckpts import strip_optimizer_states_in_folder


@define
class Args(VerboseQuietArgs):
    base_dir: Path = add_argument(
        positional=True, type=str, help="Directory to apply", default=None
    )
    glob: str = add_argument(
        "--glob", type=str, help="Glob pattern for checkpoint files", default="**/*.ckpt"
    )
    write: bool = add_argument(action="store_true", help="Confirm write.")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    target_dir = Path(args.base_dir)
    logger.info(f"Stripping optimizer states in {target_dir} matching '{args.glob}'")
    new_paths = strip_optimizer_states_in_folder(target_dir, args.glob, write=args.write)
    logger.info(f"Created {len(new_paths)} optimizer-free checkpoints")


if __name__ == "__main__":
    main()
