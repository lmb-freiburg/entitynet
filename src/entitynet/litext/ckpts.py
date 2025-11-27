"""
Modify lightning ModelCheckpoint
"""

import os
import time
from pathlib import Path
from types import new_class
from typing import Dict, List
from weakref import proxy

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor

from packg.iotools import dumps_json
from packg.log import logger
from packg.misc import format_exception
from packg.typext import PathType


class CustomModelCheckpoint(ModelCheckpoint):

    def _save_last_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
    ) -> None:
        """
        In addition to "last.ckpt", save also "last.json" with the epoch and
        global_step information. This way, this information can be accessed without loading the
        entire ckpt.
        """
        super()._save_last_checkpoint(trainer, monitor_candidates)
        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)
        if not trainer.is_global_zero:
            return
        filepath_info = Path(filepath).parent / f"{Path(filepath).stem}.json"
        metric_value = monitor_candidates[self.monitor] if self.monitor is not None else None
        info = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "metric_value": metric_value,
        }
        ckpt_str = dumps_json(info, indent=2)
        os.makedirs(filepath_info.parent, exist_ok=True)
        Path(filepath_info).write_text(ckpt_str)
        logger.debug(f"Saved last.json in {filepath_info}")

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """
        Try making the checkpointing more robust to disk or other problems.

        Every failed save fills up /tmp so we cannot just try to save every 60 seconds.
        So we wait exponentially longer each time.
        """
        sleep = 15 * 60  # initial sleep 15 minutes
        while True:
            try:
                trainer.save_checkpoint(filepath, self.save_weights_only)
                break
            except Exception as e:
                logger.error(
                    f"Error saving checkpoint to {filepath} Error {format_exception(e)} "
                    f"Will try again in {sleep} seconds"
                )
                time.sleep(sleep)
                sleep = min(sleep * 4, 16 * 3600)  # quadruple sleep time to max 16 hours

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for tlogger in trainer.loggers:
                tlogger.after_save_checkpoint(proxy(self))


def save_checkpoint_without_optimizer_states(path: PathType) -> Path:
    """
    Load a torch checkpoint, drop the optimizer states, and save it under
    "<original>-nooptstates<suffix>".
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

    new_name = f"{ckpt_path.stem}-nooptstates{ckpt_path.suffix}"
    new_path = ckpt_path.with_name(new_name)
    if new_path.is_file():
        logger.info(f"Checkpoint without optimizer states already exists at {new_path}")
        return new_path

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "optimizer_states" in checkpoint:
        del checkpoint["optimizer_states"]
    else:
        logger.warning(f"No optimizer_states found in {ckpt_path}")
    torch.save(checkpoint, new_path)
    logger.info(f"Saved checkpoint without optimizer states to {new_path}")
    return new_path


def strip_optimizer_states_in_folder(folder: PathType, pattern: str = "**/*.ckpt") -> List[Path]:
    """
    Apply optimizer state stripping to every checkpoint in a folder that matches the glob.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder {folder_path} not found")

    new_paths: List[Path] = []
    for ckpt_path in folder_path.glob(pattern):
        if ckpt_path.is_file():
            if "-nooptstates" in ckpt_path.name:
                logger.info(f"Skipping checkpoint already without optimizer states: {ckpt_path}")
                continue
            new_paths.append(save_checkpoint_without_optimizer_states(ckpt_path))
    if not new_paths:
        logger.warning(f"No checkpoints matched pattern '{pattern}' in {folder_path}")
    return new_paths
