"""
BaseModel that should handle all the basic tasks like parameters, optimizing
"""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning as lit
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from loguru import logger
from natsort import natsorted
from torch import nn

from packg import format_exception
from packg.iotools import dump_json
from packg.strings import format_pseudo_table
from visiontext.torchutils import group_params_and_data_for_display, show_param_groups_dict

from entitynet.config.main_config import Config, OptimizerCfg
from entitynet.litext.distributed_gathering import save_outputs
from entitynet.optimizers import create_optimizer_from_config
from entitynet.results.metrics_formatter import format_metric
from entitynet.schedulers import create_scheduler
from entitynet.tasks.base_task import BaseTask


class LitBaseModel(lit.LightningModule, ABC):
    config: Config
    model: nn.Module

    def __init__(self):
        super().__init__()
        self.train_task: BaseTask | None = None

    @abstractmethod
    def get_param_groups_dict(self) -> dict[str, dict]:
        """Get parameter groups for optimizer configuration.

        Returns:
            Dictionary mapping group names to parameter group configurations.
            Each group config should contain:
            - "params": list of parameters
            - "param_names": list of parameter names
            - "lr": learning rate (optional)
            - "weight_decay": weight decay (optional)
        """

    def show_param_groups_dict(self, param_groups_dict: dict):
        show_param_groups_dict(param_groups_dict)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt_cfg: OptimizerCfg = self.config.optimizer
        param_groups_dict = self.get_param_groups_dict()
        if self.config.trainer.print_param_groups:
            show_clip_param_groups_dict(param_groups_dict)

        # remove parameter names
        for group in param_groups_dict.values():
            del group["param_names"]

        # remove group names
        param_groups = list(param_groups_dict.values())
        opt = create_optimizer_from_config(param_groups, opt_cfg)

        steps_per_epoch = None
        if self.trainer.max_epochs is None:
            # TODO training step-based instead of epoch-based was never tested and will likely break
            total_steps = self.trainer.max_steps
            total_steps_for_scheduler = total_steps
            assert (
                self.config.trainer.max_epochs_for_scheduler is None
            ), f"Cannot set max_epochs_for_scheduler if max_epochs is not set."
        else:
            steps_per_epoch, total_steps, total_steps_for_scheduler = (
                self.calculate_steps_from_epochs()
            )
            if steps_per_epoch < self.config.trainer.accum_steps:
                raise ValueError(
                    f"Dataset size and batch size results in {steps_per_epoch=} which is smaller "
                    f"than {self.config.trainer.accum_steps=}. So optimizer.step() will never be "
                    f"reached, therefore no checkpoint will be written and the validation code "
                    f"will fail."
                )
        # calculate lr scheduler
        if opt_cfg.warmup_steps is None:
            if opt_cfg.warmup_epochs is None:
                # default is no warmup
                opt_cfg.warmup_steps = 0
            else:
                # calculate number of warmup steps based on number of warmup epochs
                if self.trainer.max_epochs is None:
                    raise NotImplementedError(
                        f"{self.trainer.max_epochs=}, setting max_steps, f"
                        "and setting warmup_epochs is not implemented"
                    )
                opt_cfg.warmup_steps = opt_cfg.warmup_epochs * steps_per_epoch  # type: ignore
                logger.info(f"{opt_cfg.warmup_epochs=} -> calculated {opt_cfg.warmup_steps=}")
        elif opt_cfg.warmup_epochs is not None:
            raise RuntimeError(
                f"Misconfig: Both {opt_cfg.warmup_steps=} and {opt_cfg.warmup_epochs=} are set"
            )

        if opt_cfg.constant_steps is None:
            if opt_cfg.constant_epochs is None:
                # default is no constant steps
                opt_cfg.constant_steps = 0
            else:
                # calculate number of constant steps based on number of constant epochs
                opt_cfg.constant_steps = opt_cfg.constant_epochs * steps_per_epoch  # type: ignore
                logger.info(f"{opt_cfg.constant_epochs=} -> calculated {opt_cfg.constant_steps=}")
        elif opt_cfg.constant_steps is not None:
            raise RuntimeError(
                f"Misconfig: Both {opt_cfg.constant_steps=} and {opt_cfg.constant_epochs=} are set"
            )

        logger.info(
            f"Create scheduler with expected total number of steps {total_steps_for_scheduler}"
        )
        scheduler = create_scheduler(
            opt,
            opt_cfg.scheduler_name,
            total_steps_for_scheduler,
            opt_cfg.warmup_steps,
            opt_cfg.constant_steps,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                # "monitor": self.config.trainer.ckpt.monitor,  # only needed for ReduceLROnPlateau etc.
                # "strict": True,
            },
        }

    def calculate_steps_from_epochs(self) -> int:
        # calculate the number of steps per epoch here
        batch_size = self.config.trainer.batch_size
        world_size = self.trainer.world_size

        # get accum_steps and verify the config
        if self.automatic_optimization:
            accum_steps = self.trainer.accumulate_grad_batches
            if self.config.trainer.accum_steps > 1:
                raise ValueError(
                    f"Misconfiguration: {self.automatic_optimization=} so manual accum_steps "
                    f"should be set to 1 but is: {self.config.trainer.accum_steps=}"
                )
        else:
            accum_steps = self.config.trainer.accum_steps
            if self.trainer.accumulate_grad_batches > 1:
                raise ValueError(
                    f"Misconfiguration: {self.automatic_optimization=} so the automatic "
                    f"accum_steps should be set to 1 but is: "
                    f"{self.trainer.accumulate_grad_batches=}"
                )

        global_batch_size = accum_steps * batch_size * world_size
        steps_per_epoch = len(self.train_dataset) // global_batch_size
        total_steps = steps_per_epoch * self.trainer.max_epochs
        total_steps_for_scheduler = total_steps
        if self.config.trainer.max_epochs_for_scheduler is not None:
            total_steps_for_scheduler = (
                self.config.trainer.max_epochs_for_scheduler * steps_per_epoch
            )
        logger.info(
            f"=======================================================\n"
            f"{len(self.train_dataset)=} {world_size=} {batch_size=} {accum_steps=}\n"
            f"{global_batch_size=} {steps_per_epoch=} {total_steps=} "
            f"{total_steps_for_scheduler=}\n"
            f"======================================================="
        )
        return steps_per_epoch, total_steps, total_steps_for_scheduler

    def training_step_log_lr(self):
        for i_opt, opt in enumerate(self.trainer.optimizers):
            i_opt_str = "" if i_opt == 0 else f"_o{i_opt}"
            for i_pg, pg in enumerate(opt.param_groups):
                i_pg_str = "" if i_pg == 0 else f"_g{i_pg}"
                self.log(f"lr{i_opt_str}{i_pg_str}", pg["lr"])
                # print(f"Step {self.global_step} LR {pg['lr']:.6f}")

    def setup_train_task(self, task: BaseTask, train_dataset):
        self.train_task = task
        self.train_dataset = train_dataset

    def setup_validation_tasks(self, tasks: list[BaseTask], val_datasets):
        self.val_tasks = tasks
        self.val_datasets = val_datasets

    def setup_test_tasks(self, tasks: list[BaseTask], val_datasets):
        self.test_tasks = tasks
        self.test_datasets = val_datasets

    def get_eval_output_file(self, task_key: str, suffix: str = "outputs.pt"):
        return get_eval_output_file(
            self.config.trainer.output_dir, self.eval_phase, self.epoch_identifier, task_key, suffix
        )

    def on_train_epoch_start(self) -> None:
        # for deterministic shard shuffling the webdataset loader needs to know the current epoch,
        # to allow for a different but deterministic shuffle each epoch.
        if hasattr(self.train_dataset, "shared_epoch"):
            # print_with_rank(f"Set current epoch on dataloader: {self.current_epoch}")
            self.train_dataset.shared_epoch.set_value(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        """See on_test_epoch_start documentation. Same thing applies here."""
        # update epoch identifier to save outputs with the correct epoch number in the filename
        self.epoch_identifier = f"{self.trainer.current_epoch}-{self.global_step}"
        self.eval_phase = "val"
        self._eval_finished = []

    def run_eval_step(
        self,
        batch: dict,
        batch_idx: int,
        task: BaseTask,
        output_list: list,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Run a single evaluation step. Logic is delegated to the tasks, which then make assumptions
        about the model and the batch.

        Usually this should not need to be overridden.

        Args:
            batch: Dictionary containing the batch data
            batch_idx: Index of the current batch
            task: The evaluation task being run
            output_list: List to append step outputs to
            dataloader_idx: Index of the current dataloader (default: 0)
        """
        result = task.run_eval_step(self, batch)
        output_list.append(result)

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        task = self.val_tasks[dataloader_idx]
        dataset = self.val_datasets[dataloader_idx]
        if batch_idx == 0:  # start of val task
            self.val_output = []
            self._eval_steps = get_total_val_batches(self.trainer, dataloader_idx)
            logger.debug(f"Starting validation {batch_idx} {dataloader_idx}")
            task.on_eval_start(self, dataset)
        # run single val task step
        self.run_eval_step(batch, batch_idx, task, self.val_output, dataloader_idx)
        if batch_idx == self._eval_steps - 1:  # end of val task
            self._eval_finished.append(True)
            self.run_eval_epoch_end(task, self.val_output, dataset)
        if batch_idx >= self._eval_steps:  # sanity check
            raise RuntimeError(f"{batch_idx=} {self._eval_steps - 1=} {dataloader_idx=} {task=}")

    def on_validation_epoch_end(self) -> None:
        if len(self._eval_finished) != len(self.val_tasks):  # sanity check all tasks are done
            raise RuntimeError(f"{len(self._eval_finished)=} {len(self.val_tasks)=}")
        logger.debug(f"Done testing {len(self.val_tasks)} tasks")

    def on_test_epoch_start(self) -> None:
        # at this point epoch_identifier must be already set depending on which weights are loaded
        logger.info(f"Run test for epoch_identifier={self.epoch_identifier}")
        self.eval_phase = "test"
        self._eval_finished = []

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        """
        Note: we cannot use "on_test_epoch_start" and "..._end" to start and stop the task,
        because that would start all tasks, then run all tasks, then stop all tasks.
        However each task requires significant memory to store embeddings, outputs etc. and
        this will run oom. So instead we start on batch_idx 0 and end on batch_idx -1 here.
        """
        task = self.test_tasks[dataloader_idx]
        dataset = self.test_datasets[dataloader_idx]
        if batch_idx == 0:  # start of test task
            self.test_output = []
            self._eval_steps = get_total_test_batches(self.trainer, dataloader_idx)
            logger.debug(f"Starting test task {dataloader_idx}: {task} {self._eval_steps=}")
            task.on_eval_start(self, dataset)
        # run single test task step
        self.run_eval_step(batch, batch_idx, task, self.test_output, dataloader_idx)
        if batch_idx == self._eval_steps - 1:  # end of test task
            self._eval_finished.append(True)
            self.run_eval_epoch_end(task, self.test_output, dataset)
        if batch_idx >= self._eval_steps:  # sanity check
            raise RuntimeError(f"{batch_idx=} {self._eval_steps - 1=} {dataloader_idx=} {task=}")

    def on_test_epoch_end(self) -> None:
        if len(self._eval_finished) != len(self.test_tasks):  # sanity check all tasks are done
            raise RuntimeError(f"{len(self._eval_finished)=} {len(self.test_tasks)=}")
        logger.debug(f"Done testing {len(self.test_tasks)} tasks")

    def run_eval_epoch_end(self, task: BaseTask, output: list[Any], dataset: Any):
        metric_prefix = self.eval_phase
        task_key = task.task_key
        if len(output) == 0:
            # some validation tasks may not have any outputs
            output_dict = None
        else:
            # in case there are outputs, we must gather, deduplicate and save them
            # TODO: this all gather uses GPU and can run OOM easily
            #       convert tensors to json and use gather_object_on_filesystem
            output_dict = save_outputs(
                self.trainer,
                output,
                self.all_gather,
                self.get_eval_output_file(task_key, suffix="outputs.pt"),
                self.get_eval_output_file(task_key, suffix="extras.pt"),
                task.extra_output_keys,
            )
        metrics_dict = task.on_eval_end(self, dataset, output_dict)

        # if there are any metrics, they will be aggregated on rank 0.
        # on all other ranks metrics_dict is None.
        # these are end-of-epoch scalar metrics (already aggregated) so we set batch_size=1.
        if metrics_dict is not None and self.trainer.is_global_zero:
            results_dict, results_dict_nan_none = {}, {}
            for k, v in metrics_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                full_key = f"{metric_prefix}/{task_key}_{k}"
                results_dict[full_key] = v
                results_dict_nan_none[full_key] = v
                if math.isnan(v) or np.isnan(v):
                    results_dict_nan_none[full_key] = None
            # print(f"Logging dict of len {results_dict} for task {task_key}")
            # lightning logger crashes if None values are given, but accepts NaN
            for k, v in results_dict.items():
                try:
                    self.log(
                        name=k,
                        value=v,
                        on_step=False,
                        on_epoch=True,
                        rank_zero_only=True,
                        add_dataloader_idx=False,
                        batch_size=1,
                    )
                except Exception as e:
                    logger.error(
                        f"Logging failed for task {task_key} {format_exception(e)}\n" f"{k=} {v=}"
                    )
            # json does not support NaN but supports None
            json_file = self.get_eval_output_file(task_key, suffix="results.json")
            dump_json(results_dict_nan_none, json_file, verbose=False, create_parent=True)
        self.trainer.strategy.barrier()
        if self.trainer.is_global_zero:
            metrics_strs = [f"{k}={format_metric(k, v)}" for k, v in metrics_dict.items()]
            logger.info(
                f"Epoch {self.current_epoch:2d} Task {task_key}\n{format_pseudo_table(metrics_strs)}"
            )

        # print_with_rank(f"Eval epoch done: {self.current_epoch}")  # here world_size is still 8

    def check_for_nans(self):
        count_nans_total = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                count_nans = torch.isnan(param.grad).sum().item()
                if count_nans > 0:
                    logger.debug(f"{count_nans} NaN gradient detected in {name}")
                    count_nans_total += count_nans
        if count_nans_total > 0:
            logger.warning(f"{self.global_step=} {count_nans_total=}")

    def print_params_grads(self):
        params_counts = defaultdict(list)
        params_names = defaultdict(list)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                group_name = "grad"
            else:
                group_name = "no_grad"
            params_counts[group_name].append(list(param.shape))
            params_names[group_name].append(name)
        for group_name, counted_params in params_counts.items():
            print(
                f"Group: {group_name}, number of parameters: {len(counted_params)} "
                f"total size: {int(sum([np.prod(p) for p in counted_params])):_d}"
            )

    def get_trainer_if_attached(self) -> lit.Trainer | None:
        try:
            trainer = self.trainer
        except RuntimeError:  # RuntimeError: LitOpenClip is not attached to a `Trainer`.
            trainer = None
        return trainer


def get_eval_output_file(output_dir, eval_phase, epoch_identifier, task_key, suffix):
    return (
        Path(output_dir)
        / "outputs"
        / f"{eval_phase}_ckpt-{epoch_identifier}_task-{task_key}-{suffix}"
    )


def get_total_val_batches(trainer: Trainer, dataloader_idx: int | None) -> int:
    """
    Source: site-packages/lightning/pytorch/callbacks/progress/progress_bar.py
        ProgressBar.total_val_batches_current_dataloader
    """
    batches = trainer.num_sanity_val_batches if trainer.sanity_checking else trainer.num_val_batches
    if isinstance(batches, list):
        assert dataloader_idx is not None
        n_batches = batches[dataloader_idx]
    else:
        n_batches = batches
    assert not math.isinf(n_batches), f"Infinite val dataloader not supported. {dataloader_idx=}"
    return n_batches


def get_total_test_batches(trainer: Trainer, dataloader_idx: int | None) -> int:
    """
    Source: site-packages/lightning/pytorch/callbacks/progress/progress_bar.py
        ProgressBar.total_test_batches_current_dataloader
    """
    batches = trainer.num_test_batches
    if isinstance(batches, list):
        assert dataloader_idx is not None
        n_batches = batches[dataloader_idx]
    else:
        n_batches = batches
    assert not math.isinf(n_batches), f"Infinite val dataloader not supported. {dataloader_idx=}"
    return n_batches


def show_clip_param_groups_dict(param_groups_dict):
    n_params_by_tower = defaultdict(int)
    for group_name, group_content in natsorted(param_groups_dict.items(), key=lambda x: x[0]):
        params = group_content["params"]
        param_names = group_content["param_names"]
        wd = group_content["weight_decay"]
        lr = group_content["lr"]
        logger.info(f"{group_name:20s} {lr=:7.1e} {wd=:7.1e}")
        param_dict = {param_name: param for param_name, param in zip(param_names, params)}
        param_shapes = []
        for param_name, param in natsorted(param_dict.items(), key=lambda x: x[0]):
            param_shape = tuple(param.shape)
            param_shapes.append(param_shape)
            n_params = np.prod(param_shape)
            if param_name.startswith("visual."):
                n_params_by_tower["visual"] += n_params
            else:
                n_params_by_tower["other"] += n_params
        new_names, new_data = group_params_and_data_for_display(param_names, param_shapes)
        new_shapes = [d.shape for d in new_data]
        for param_name, param_shape in zip(new_names, new_shapes):
            logger.info(f"    {str(param_shape):20s} {param_name}")
        logger.info("")
    for tower, n_params in n_params_by_tower.items():
        logger.info(f"Number of parameters in {tower}: {int(n_params):_d}")
