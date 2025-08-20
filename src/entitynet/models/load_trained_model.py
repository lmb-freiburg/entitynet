"""
Load trained models from checkpoints.

This module provides functionality to load trained CLIP models from checkpoints
created during training.
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from entitynet.config.config_factory import load_config_from_file
from entitynet.models.model_factory import build_model_from_config
from entitynet.models.tokenizer_factory import build_tokenizer_from_config
from entitynet.preprocessor.preprocessor_factory import build_vis_preprocessor_from_config
from entitynet.results.checkpoint_finder import find_checkpoints


def load_trained_model(
    config_file: Path, ckpt_mode: str = "last", run_id: Optional[str] = None, device: str = "cpu"
):
    """
    Load a trained model from checkpoint.

    Args:
        config_file: Path to the experiment config file
        load_best: If True, load the best checkpoint. If False, load the last checkpoint
        run_id: Optional run ID for the experiment subfolder
        device: Device to load the model on ("cpu", "cuda", etc.)

    Returns:
        Loaded model with weights from checkpoint,
        preprocessor for visual inputs,
        tokenizer for text inputs,
        config object
    """
    config = load_config_from_file(config_file)

    preproc_cfg = config.model.vis_preproc
    vis_prep = build_vis_preprocessor_from_config(preproc_cfg)
    tokenizer = build_tokenizer_from_config(config.model)

    trcfg = config.trainer
    trcfg.output_dir = Path(trcfg.output_dir)
    if run_id is not None:
        trcfg.output_dir = trcfg.output_dir / run_id
    else:
        trcfg.output_dir = trcfg.output_dir / "defaultrun"
    ckpt_dir = trcfg.output_dir / "ckpt"

    # ----- create model
    logger.info(f"Creating model from config: {config_file}")
    model = build_model_from_config(config)
    model.config = config

    # ----- find and load checkpoint
    last_ckpt, best_ckpt, _ = find_checkpoints(ckpt_dir)

    if ckpt_mode == "best":
        ckpt = best_ckpt
        logger.info("Loading best checkpoint")
        assert best_ckpt is not None, f"No best checkpoint found in {ckpt_dir}"
    elif ckpt_mode == "last":
        ckpt = last_ckpt
        logger.info("Loading last checkpoint")
        assert last_ckpt is not None, f"No last checkpoint found in {ckpt_dir}"
    elif ckpt_mode == "none":
        ckpt = None
        logger.info("Not loading any checkpoint (use pretrained weights)")
    else:
        raise ValueError(f"Invalid checkpoint mode: {ckpt_mode}")

    # ----- load checkpoint
    if ckpt is not None:
        ckpt_file = ckpt["file"]
        logger.info(f"Checkpoint: {ckpt_file} epoch: {ckpt['epoch']}, step: {ckpt['global_step']}")
        ckpt_content = torch.load(ckpt_file, map_location=device, weights_only=True)
        # this is a lightning saved checkpoint so it should 100% fit the model
        model.load_state_dict(ckpt_content["state_dict"], strict=True)

    # ----- move model to device
    model = model.to(device)
    model.eval()

    logger.info(f"Successfully loaded model from checkpoint")
    logger.info(f"Model device: {next(model.parameters()).device}")

    return model, vis_prep, tokenizer, config
