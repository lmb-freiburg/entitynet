from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
from torch import Tensor

from entitynet.config.task_config import BaseTaskCfg

EVAL_OUTPUT_TYPE = dict[str, Union[Tensor, Any]] | None


class BaseTask(ABC):
    def __init__(self, task_key: str, task_cfg: BaseTaskCfg):
        self.task_key: str = task_key
        self.task_cfg: BaseTaskCfg = task_cfg
        self.extra_output_keys = []
        self.setup()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.task_key})"

    def setup(self):
        pass

    def on_eval_start(self, model, dataset):
        pass

    @abstractmethod
    def run_eval_step(self, model, batch: dict):
        pass

    def on_eval_end(self, model, dataset, eval_output: EVAL_OUTPUT_TYPE) -> dict[str, float]:
        pass

    def aggregate_loss_if_exists(self, model, dataset, eval_output: EVAL_OUTPUT_TYPE):
        if eval_output is None:
            return None
        if "loss" in eval_output:
            # it should be a list of floats with one float per datapoint
            loss_list = eval_output["loss"]
            # print(f"Got loss_list type {type(loss_list)} with {len(loss_list)} elements")
            if not model.trainer.sanity_checking:
                if len(loss_list) != len(dataset):
                    raise RuntimeError(f"{len(loss_list)=} != {len(dataset)=}")
            return np.mean(loss_list)
        return None
