import torch
from loguru import logger

from entitynet.config.main_config import OptimizerCfg


def create_optimizer_from_config(params, opt_cfg: OptimizerCfg):
    opt_name = opt_cfg.opt_name
    logger.info(f"Creating optimizer {opt_name} with params {opt_cfg.hparams}")
    if opt_cfg.opt_name == "adamw":
        opt = torch.optim.AdamW(params, **opt_cfg.hparams)
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.opt_name}")
    return opt
