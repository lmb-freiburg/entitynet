"""
Utility to cleanup the outputs of the training that are not needed anymore.
"""

import re
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from loguru import logger

from packg import format_exception
from packg.typext import PathType
from visiontext.distutils import WorldInfo

from entitynet.config.main_config import CleanupKeepOutputC
from entitynet.results.checkpoint_finder import find_checkpoints


class CleanupOutputsCallback(Callback):
    def __init__(self, output_dir: PathType, keep_output_mode: str = CleanupKeepOutputC.CKPT):
        self.output_dir = Path(output_dir)
        self.keep_output_mode = keep_output_mode

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        cleanup_val_outputs(
            self.output_dir, self.keep_output_mode, current_epoch=trainer.current_epoch
        )


RE_OUTPUT = re.compile(r"val_ckpt-(\d+)-(\d+)_task-(.*)\..*")


def cleanup_val_outputs(
    experiment_dir,
    keep_output_mode=CleanupKeepOutputC.CKPT,
    write=True,
    current_epoch: int | None = None,
    trainer=None,
) -> list[Path]:
    world_info = WorldInfo(trainer)
    if not world_info.is_global_zero:
        return []
    if keep_output_mode == CleanupKeepOutputC.ALL:
        return []
    files_to_delete = []
    val_outputs = list((experiment_dir / "outputs").glob("val_*outputs.pt"))
    if keep_output_mode == CleanupKeepOutputC.NONE:
        for file in val_outputs:
            files_to_delete.append(file)
            if write:
                file.unlink()
        return files_to_delete

    if keep_output_mode == CleanupKeepOutputC.CKPT:
        # only save outputs for which also checkpoints were saved
        try:
            _, _, all_ckpts = find_checkpoints(experiment_dir / "ckpt")
        except FileNotFoundError as e:
            # do not crash the entire training if something goes wrong here
            logger.error(f"Could not find checkpoints in {experiment_dir} - {format_exception(e)}")
            return []
        epochs = [ckpt["epoch"] for ckpt in all_ckpts]
        epochs_with_ckpt_set = set(epochs)

        for val_output in val_outputs:
            name = val_output.name
            match = RE_OUTPUT.match(name)
            if match is None:
                logger.error(f"Could not match name {name} with regex {RE_OUTPUT}")
            epoch, global_step, task = match.groups()
            epoch, global_step = int(epoch), int(global_step)
            if epoch in epochs_with_ckpt_set:
                # checkpoint exists, so keep the output.
                continue
            if current_epoch is not None and epoch >= current_epoch:
                # do not delete outputs from the current epoch or later, delete only past stuff
                continue
            logger.debug(
                f"Deleting {val_output} since it is not in epochs to keep: "
                f"{sorted(epochs_with_ckpt_set)} and < {current_epoch=} if that is set."
            )
            files_to_delete.append(val_output)
            if write:
                val_output.unlink()
        return files_to_delete

    raise ValueError(f"Invalid {keep_output_mode=}")
