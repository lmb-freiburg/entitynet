import numpy as np
import torch
from torch.nn import functional as F


try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False


def get_init_logits_for_loss_name(loss_name: str) -> tuple[float, float]:
    if loss_name == "clip":
        init_logit_scale = 2.659260036932778  # np.log(1 / 0.07)
        init_logit_bias = None
    else:
        # siglip style models need different init
        init_logit_scale = np.log(10)
        init_logit_bias = -10
    return init_logit_scale, init_logit_bias
