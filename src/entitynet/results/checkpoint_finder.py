import shutil
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from packg import Const
from packg.iotools import dump_json, load_json
from packg.typext import PathType
from visiontext.distutils import WorldInfo


class CkptBehaviorC(Const):
    """What to do if a checkpoint exists for the same experiment."""

    RESUME = "resume"
    SKIP = "skip"
    REMOVE = "remove"
    # another option could be to automatically increment the run_id and start a new experiment


def find_ckpt_to_resume(
    output_dir: PathType,
    behavior: str = CkptBehaviorC.RESUME,
) -> tuple[str | None, int, int]:
    """
    Find the checkpoint to resume training from.

    Args:
        output_dir: experiment output directory
        behavior: what to do if checkpoint found (default resume)

    Returns:
        either checkpoint file string, checkpoint epoch, checkpoint step or
        None, -1, -1 if no checkpoint found

    """
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "ckpt"
    return_not_found = None, -1, -1
    if not output_dir.is_dir():
        return return_not_found
    last_ckpt, best_ckpt, _ = find_checkpoints(ckpt_dir)
    logger.info(f"Found {last_ckpt=}")
    logger.info(f"Found {best_ckpt=}")
    if behavior == "resume":
        if last_ckpt is not None:
            logger.info(f"Will resume training from {last_ckpt}")
            r_ckpt_file = last_ckpt["file"]
            r_epoch = last_ckpt["epoch"]
            r_step = last_ckpt["global_step"]
            return r_ckpt_file, r_epoch, r_step
        logger.warning(f"No last ckpt found in {ckpt_dir}. Running from scratch.")
        return return_not_found
    if behavior == "skip":
        raise FileExistsError(f"Output folder exists: {output_dir}. Behaviour is to skip.")
    if behavior == "remove":
        logger.warning(f"Removing existing output folder: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
        return return_not_found
    raise ValueError(f"Unknown behaviour: {behavior}")


def find_checkpoints(
    ckpt_dir: PathType, log_errors: bool = True
) -> tuple[dict | None, dict | None, list[dict]]:
    ckpt_dir: Path = Path(ckpt_dir)
    last_ckpt_file = ckpt_dir / "last.ckpt"
    last_ckpt_nos_file = ckpt_dir / "last-nooptstates.ckpt"
    if last_ckpt_nos_file.is_file() and not last_ckpt_file.is_file():
        last_ckpt_file = last_ckpt_nos_file
    if last_ckpt_file.is_file():
        last_json_file = ckpt_dir / "last.json"
        if not last_json_file.is_file():
            # in rare cases (old experiments, last.json got deleted manually, bugs...) the last.json
            # file is missing. instead of crashing, recover the information from the last.ckpt file.
            logger.error(f"last.ckpt found but last.json missing in {ckpt_dir}")
            last_ckpt_info = convert_last_ckpt_to_last_json(last_ckpt_file)
            if WorldInfo().is_global_zero:
                dump_json(last_ckpt_info, last_json_file, indent=2)
        else:
            last_ckpt_info = load_json(ckpt_dir / "last.json")

        if "metric_value" not in last_ckpt_info and log_errors:
            logger.error(
                f"last.json does not contain metric_value (old version of checkpoint): "
                f"{last_ckpt_info} in {last_ckpt_file}"
            )

        last_ckpt = {
            "file": last_ckpt_file.as_posix(),
            "epoch": int(last_ckpt_info["epoch"]),
            "global_step": int(last_ckpt_info["global_step"]),
            "metric_value": last_ckpt_info.get("metric_value"),
        }
    else:
        last_ckpt = None

    # load other checkpoint files, prefering with opt state over without opt state.
    other_ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    # Remove last.ckpt files and create dict with opt state preference
    other_ckpts_dict = {}
    for ckpt in other_ckpts:
        if ckpt.name in {"last.ckpt", "last-nooptstates.ckpt"}:
            continue
        # Extract base name (without -nooptstates suffix)
        base_name = ckpt.name.replace("-nooptstates", "")
        has_opt_states = "-nooptstates" not in ckpt.name
        # Keep ckpt with opt states if available, otherwise keep without
        if base_name not in other_ckpts_dict:
            other_ckpts_dict[base_name] = ckpt
        elif has_opt_states:
            # Replace with version that has opt states
            other_ckpts_dict[base_name] = ckpt
    other_ckpts = sorted(other_ckpts_dict.values(), key=lambda p: p.name)
    print(f"{other_ckpts=}")

    all_ckpts = []
    mode_final = None
    metric_name_final = None
    for other_ckpt_file in other_ckpts:
        if other_ckpt_file.name in set(["last.ckpt", "last-nooptstates.ckpt"]):
            continue
        msplits = other_ckpt_file.name.replace("-nooptstates", "").split("-")
        if len(msplits) == 5:
            # e.g. 10-7645-min-val_loss-0.810585.ckpt
            epoch, global_step, mode, metric_name, metric_value = msplits
            metric_value = float(metric_value[:-5])
        elif len(msplits) == 6:
            # e.g. 10-7645-min-val_loss--0.810585.ckpt for negative metric value
            epoch, global_step, mode, metric_name, blank, pos_metric_value = msplits
            assert blank == ""
            metric_value = -float(pos_metric_value[:-5])
        else:
            raise ValueError(f"Unexpected checkpoint filename: {other_ckpt_file.name}")
        if mode_final is None:
            mode_final = mode
        else:
            assert mode_final == mode, f"Mode mismatch: {mode_final} vs {mode}"
        if metric_name_final is None:
            metric_name_final = metric_name
        else:
            assert (
                metric_name_final == metric_name
            ), f"Metric name mismatch: {metric_name_final} vs {metric_name}"

        all_ckpts.append(
            {
                "file": other_ckpt_file.as_posix(),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "metric_value": float(metric_value),
            }
        )
    epochs_set = set([ckpt["epoch"] for ckpt in all_ckpts])
    if last_ckpt is not None:
        last_epoch = last_ckpt["epoch"]
        if last_epoch not in epochs_set:
            all_ckpts.append(last_ckpt)

    all_ckpts_with_metric = [ckpt for ckpt in all_ckpts if ckpt["metric_value"] is not None]
    if len(all_ckpts_with_metric) > 0:
        metric_values = [ckpt["metric_value"] for ckpt in all_ckpts_with_metric]
        sort = np.argsort(metric_values)
        if mode_final == "min":
            best_idx = sort[0]
        elif mode_final == "max":
            best_idx = sort[-1]
        else:
            raise ValueError(f"Unknown mode: {mode_final}")
        best_ckpt = all_ckpts_with_metric[best_idx]
    else:
        best_ckpt = None
    return last_ckpt, best_ckpt, all_ckpts


def convert_last_ckpt_to_last_json(ckpt_file: Path) -> dict:
    ckpt_file = Path(ckpt_file)
    ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    epoch = ckpt_data["epoch"]
    global_step = ckpt_data["global_step"]
    # so it seems not possible to get the most up to date value. but it's probably not
    # necessary because the last.ckpt contains everything for the callback to work properly.
    current_score = None
    last_json_content = {
        "epoch": epoch,
        "global_step": global_step,
        "metric_value": current_score,
    }
    return last_json_content
