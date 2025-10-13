"""
Config is a nested class structure where all experiment config yamls will be mapped to.

The main config will load it's children depending on their factory. E.g.: if the training task is
contrastive, it will parse the "train_task" field as a ClipContrastiveTaskCfg.
Which is a subclass of BaseTaskCfg.

The reason for this: Otherwise, all fields for all kinds of tasks and models etc. must
be present in the main config, making it very large.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attr import asdict
from attrs import define

from packg import Const
from packg.iotools import dumps_yaml

from entitynet.config.model_config import BaseModelCfg
from entitynet.config.task_config import BaseTaskCfg


@define(auto_attribs=True, kw_only=True)
class Config:
    model: BaseModelCfg = None
    trainer: TrainerCfg = None
    eval_tasks: dict[str, BaseTaskCfg] | None = None  # used to define new eval tasks on the fly
    train_task: BaseTaskCfg | None = None
    optimizer: OptimizerCfg | None = None

    def __repr__(self):
        return f"{dumps_yaml(asdict(self), standard_format=False)}"


@define(auto_attribs=True, kw_only=True, slots=False)
class OptimizerCfg:
    opt_name: str = None
    hparams: dict[str, Any] = None
    warmup_steps: int | None = None
    warmup_epochs: int | float | None = None
    constant_steps: int | None = None
    constant_epochs: int | float | None = None
    scheduler_name: str = "const"
    # options in case layer lr decay is used (different lr per layer)
    vision_layer_decay_factor: float | None = 1.0
    text_layer_decay_factor: float | None = 1.0
    layer_decay_output_layer_has_highest_lr: bool = False
    clip_grad_norm: float | None = None  # max norm for gradient clipping, None to disable


@define(auto_attribs=True, kw_only=True)
class CkptCfg:
    """
    config for lightning.pytorch.callbacks.ModelCheckpoint.
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
    """

    monitor: str | None = None
    verbose: bool = False
    save_top_k: int = 2  # 0: no checkpoints, -1: all checkpoints
    mode: str = "min"
    save_weights_only: bool = False  # TODO this is ignored and always false
    every_n_train_steps: int = 0
    every_n_epochs: int = 1


class CleanupKeepOutputC(Const):
    CKPT = "ckpt"
    NONE = "none"
    ALL = "all"


@define(auto_attribs=True, kw_only=True)
class TrainerCfg:
    # others
    workers: int = 4
    output_dir: str | Path | None = None
    experiment_name: str | None = None
    project_name: str | None = None  # used for e.g. wandb
    batch_size: int = 32
    accum_steps: int = 1
    batch_size_eval: int = 32
    seed: int = 0
    val_task_keys: list[str] | str | None = None
    test_task_keys: list[str] | str | None = None
    max_steps: int = -1
    max_epochs: int = -1
    max_epochs_for_scheduler: int | None = None
    val_check_interval: int | float | None = None
    check_val_every_n_epoch: int | None = None
    log_every_n_steps: int = 50
    num_sanity_val_steps: int | None = None
    on_exists: str = "resume"  # when output directory exists: resume, skip, or remove
    ckpt: CkptCfg | None = None
    keep_output_mode: str = CleanupKeepOutputC.CKPT
    test_best: bool = True
    test_last: bool = False
    n_images_to_save: int = 128
    check_for_nans: bool = False  # check and print if there are nans, will cost performance
    print_param_groups: bool = True
    print_config: bool = True
    eval_force_output_embeddings: bool = False
    # technical settings
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
    accelerator: str = "gpu"
    devices: list[int] | int | None = None
    precision: int | str = "32-true"
    num_nodes: int = 1
    strategy: str = "auto"
    set_float32_matmul_precision: str | None = None
    log_data_locally: bool = False  # avoid logging images and text to external services
