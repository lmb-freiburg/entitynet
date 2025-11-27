from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from attrs import define
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from entitynet.results.checkpoint_finder import find_checkpoints


def _normalize_path(path: Path) -> Path:
    """Return a normalized absolute path for stable comparisons."""
    return path.resolve(strict=False)


@define
class Args(VerboseQuietArgs):
    base_dir: Path = add_argument(
        positional=True, type=str, help="Directory containing experiment folders"
    )
    glob: str = add_argument(
        type=str,
        default="**/runconfig.yaml",
        help="Glob pattern (relative to base_dir) for locating experiment runconfig files",
    )
    write: bool = add_argument(
        action="store_true",
        help="Actually delete files. Without this flag the script only prints the actions",
    )


def main() -> None:
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    experiments = list(iter_experiments(base_dir, args.glob))
    if not experiments:
        logger.warning(f"No experiments found under {base_dir} matching '{args.glob}'")
        return

    logger.info(
        f"Processing {len(experiments)} experiments under {base_dir} "
        f"({'dry-run' if not args.write else 'deleting'})"
    )
    for exp_dir in experiments:
        ckpt_dir = exp_dir / "ckpt"
        remove_extra_checkpoints(ckpt_dir, dry_run=not args.write)


def iter_experiments(base_dir: Path, pattern: str) -> Iterable[Path]:
    for runconfig in sorted(base_dir.glob(pattern)):
        if runconfig.is_file():
            yield runconfig.parent


def remove_extra_checkpoints(ckpt_dir: Path, *, dry_run: bool) -> None:
    if not ckpt_dir.is_dir():
        logger.debug(f"Skipping {ckpt_dir} (missing)")
        return

    try:
        last_ckpt, best_ckpt, _ = find_checkpoints(ckpt_dir)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"Failed to inspect checkpoints in {ckpt_dir}: {exc}")
        return

    keep_paths = set()
    if last_ckpt:
        keep_paths.add(_normalize_path(Path(last_ckpt["file"])))
        last_json = ckpt_dir / "last.json"
        if last_json.exists():
            keep_paths.add(_normalize_path(last_json))
    if best_ckpt:
        keep_paths.add(_normalize_path(Path(best_ckpt["file"])))

    entries = sorted(ckpt_dir.iterdir())
    to_remove: list[Path] = []
    for entry in entries:
        if _normalize_path(entry) not in keep_paths:
            to_remove.append(entry)

    last_name = Path(last_ckpt["file"]).name if last_ckpt else "none"
    best_name = Path(best_ckpt["file"]).name if best_ckpt else "none"
    logger.info(f"{ckpt_dir}: keeping last={last_name}, best={best_name}")

    if not to_remove:
        logger.info(f"{ckpt_dir}: nothing to delete")
        return

    for path in to_remove:
        if dry_run:
            logger.info(f"[dry-run] Would remove {path}")
            continue
        # if path.is_dir():
        #     shutil.rmtree(path)
        #     logger.info(f"Removed directory {path}")
        # else:
        path.unlink()
        logger.info(f"Removed file {path}")


if __name__ == "__main__":
    main()
