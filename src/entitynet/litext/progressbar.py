"""
Based on python3.10/site-packages/lightning/pytorch/callbacks/progress/tqdm_progress.py

Limit the maximum width of tqdm progress bars to avoid 200+ width progress bars.
Those can in some situations (screens, tmux) cause unwanted linebreaks,
they increase filesize of logs and are just generally unnecessary.
"""

import importlib.util
import sys
from typing import Any, Optional, Union

import lightning
import numpy as np
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf
from typing_extensions import override

from entitynet.tasks.base_task import BaseTask

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

_PAD_SIZE = 5
TQDM_WID = 90
_DYNAMIC_NCOLS = False


class Tqdm(_tqdm):
    """
    Change to limit the max ncols of the progressbar, avoids 200 char bars spamming the terminal.
    """

    def __init__(self, *args, max_ncols: Optional[int] = TQDM_WID, **kwargs):
        super().__init__(*args, **kwargs)
        if self.disable:
            # pbar is disabled, no need to modify ncols
            return
        if max_ncols is not None and self.ncols is not None:
            self.ncols = min(self.ncols, max_ncols)

    @staticmethod
    def format_num(n: Union[int, float, str]) -> str:
        """Add additional padding to the formatted numbers."""
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
            assert isinstance(n, str)
        if should_be_padded and "e" not in n:
            if "." not in n and len(n) < _PAD_SIZE:
                try:
                    _ = float(n)
                except ValueError:
                    return n
                n += "."
            n += "0" * (_PAD_SIZE - len(n))
        return n


class CustomTQDMProgressBar(TQDMProgressBar):
    @override
    def on_test_batch_start(
        self,
        trainer: "lightning.Trainer",
        pl_module: "lightning.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return
        task_list = []
        try:
            task_list: list[BaseTask] = pl_module.test_tasks
        except AttributeError:
            pass

        description = f"{self.test_description} DataLoader {dataloader_idx}"
        if len(task_list) > 0 and len(task_list) > dataloader_idx:
            task = task_list[dataloader_idx]
            description = f"{self.test_description} #{dataloader_idx}: {task.task_key}"

        self.test_progress_bar.reset(convert_inf(self.total_test_batches_current_dataloader))
        self.test_progress_bar.initial = 0
        self.test_progress_bar.set_description(description)

    def _should_update(self, current: int, total: int) -> bool:
        """
        To trade of between spamming the log and not updating, update
        pbar at 1, 2, 4, 8, 16, 32, 50, 100, 150, 200, ...
        """
        if not self.is_enabled:
            return False
        if current == total or current % self.refresh_rate == 0:
            return True
        if current > self.refresh_rate:
            return False
        current_log = np.log2(current)
        if np.isclose(current_log, int(current_log), atol=1e-3):
            return True
        return False

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        return Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=_DYNAMIC_NCOLS,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=_DYNAMIC_NCOLS,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        return Tqdm(
            desc=self.predict_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=_DYNAMIC_NCOLS,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=_DYNAMIC_NCOLS,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        return Tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=_DYNAMIC_NCOLS,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    @override
    def on_sanity_check_start(self, *_: Any) -> None:
        self.val_progress_bar = self.init_sanity_tqdm()
        self.train_progress_bar = Tqdm(disable=True)  # dummy progress bar
