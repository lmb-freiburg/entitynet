"""
Modify lightning ModelCheckpoint
"""

import os
import time
from pathlib import Path
from typing import Dict
from weakref import proxy

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor

from packg.iotools import dumps_json
from packg.log import logger
from packg.misc import format_exception


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
