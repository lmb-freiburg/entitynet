"""
Modify lightning EvaluationLoop:

The original loop creates a separate collection of metrics for each dataloader.
This loop merges all dataloader metrics into one and renames them based on the dataloader name.
"""

from collections import ChainMap
from typing import List

import lightning.pytorch as pl
import torch
from lightning.pytorch.loops.evaluation_loop import _OUT_DICT, _EvaluationLoop  # noqa
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from loguru import logger


class CustomEvalLoop(_EvaluationLoop):

    def __init__(
        self,
        trainer: "pl.Trainer",
        trainer_fn: TrainerFn,
        stage: RunningStage,
        verbose: bool = True,
        inference_mode: bool = True,
        reraise: bool = False,
    ) -> None:
        super().__init__(trainer, trainer_fn, stage, verbose=verbose, inference_mode=inference_mode)
        self.reraise = reraise
        logger.debug(f"Created custom eval loop!")

    def on_run_end(self) -> List[_OUT_DICT]:
        """Runs the ``_on_evaluation_epoch_end`` hook."""
        # if `done` returned True before any iterations were done, this won't have been called in `on_advance_end`
        self.trainer._logger_connector.epoch_end_reached()  # noqa
        self.trainer._logger_connector._evaluation_epoch_end()  # noqa

        # hook
        self._on_evaluation_epoch_end()

        logged_outputs, self._logged_outputs = self._logged_outputs, []  # free memory

        # include any logged outputs on epoch_end
        epochend_logged_outputs = self.trainer._logger_connector.update_eval_epoch_metrics()  # noqa
        all_logged_outputs = dict(ChainMap(*logged_outputs))  # list[dict] -> dict
        all_logged_outputs.update(epochend_logged_outputs)
        for dl_outputs in logged_outputs:
            dl_outputs.update(epochend_logged_outputs)

        # ----- changes
        # logged_outputs contains the step metrics logged individually per dataloader
        # list (len num_dataloaders) each dict {metric_key: metric_value}
        # epoch_end_logged_outputs contains the already aggregated epoch outputs
        # dict {metric_key: metric_value}

        logger.warning(f"{type(self).__name__}: Aggregating of all dataloader metrics into one.")
        # problem is now that the end of epoch metrics get logged 5 times for the 5 dataloaders
        # the easy fix is to just aggregate everything again here.
        logged_outputs_single = {}
        for dl_outputs in logged_outputs:
            for k, v in dl_outputs.items():
                # remove dataloader idx (_print_results would remove it anyway)
                k = k.split("/dataloader_idx_")[0]
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if k not in logged_outputs_single:
                    logged_outputs_single[k] = v
                else:
                    old_v = logged_outputs_single[k]
                    delta = abs(v - old_v)
                    if delta > 1e-6:
                        raise ValueError(
                            f"Error: Metric {k} was logged by different dataloaders and got values "
                            f"{v=} {old_v=} {delta=}. This is not supported. Add a unique key "
                            f"to the metric."
                        )
        logged_outputs = [logged_outputs_single]

        # log metrics
        try:
            self.trainer._logger_connector.log_eval_end_metrics(all_logged_outputs)  # noqa
        except ValueError as e:
            logger.error(f"{all_logged_outputs}")
            logger.error(f"Error logging metrics: {e}")
            if self.reraise:
                raise e

        # hook
        self._on_evaluation_end()

        # enable train mode again
        self._on_evaluation_model_train()

        if self.verbose and self.trainer.is_global_zero:
            self._print_results(logged_outputs, self._stage.value)

        return logged_outputs
