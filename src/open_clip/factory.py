import json
import os
import re
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from loguru import logger

from .coca_model import CoCa
from .model import (
    CLIP,
    CustomTextCLIP,
    convert_to_custom_text_state_dict,
    convert_weights_to_lp,
    get_cast_dtype,
    resize_pos_embed,
    resize_text_pos_embed,
    set_model_preprocess_cfg,
)
from .pretrained import (
    download_pretrained,
    download_pretrained_from_hf,
    get_pretrained_cfg,
    list_pretrained_tags_by_model,
)
from .tokenizer import DEFAULT_CONTEXT_LENGTH, HFTokenizer, SimpleTokenizer
from .transform import (
    AugmentationCfg,
    PreprocessCfg,
    image_transform_v2,
    merge_preprocess_dict,
    merge_preprocess_kwargs,
)

HF_HUB_PREFIX = "hf-hub:"
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


# Define Schema Prefixes as constants
HF_HUB_PREFIX = "hf-hub:"
LOCAL_DIR_PREFIX = "local-dir:"


def parse_model_name(model_name: str) -> Tuple[Optional[str], str]:
    """
    Parses a model name string to identify a schema and the remaining identifier.

    Args:
        model_name: The model name string (e.g., 'ViT-B-32',
                    'hf-hub:org/repo', 'local-dir:/path/to/dir',
                    'local-dir:./relative/path').

    Returns:
        A tuple (schema, identifier):
          - schema (Optional[str]): 'hf-hub', 'local-dir', or None if no schema detected.
          - identifier (str): The part after the schema prefix, or the original
                              string if no schema was present. For 'local-dir',
                              this is the raw path string provided.
    Raises:
        ValueError: If a schema prefix is present but the identifier part is empty.
    """
    # Check for local directory schema first
    if model_name.startswith(LOCAL_DIR_PREFIX):
        # Extract the identifier (path) after the prefix
        identifier = model_name[len(LOCAL_DIR_PREFIX) :]
        # Validate that the identifier (path) is not empty
        if not identifier:
            raise ValueError("Empty path specified after 'local-dir:' schema.")
        # Return the schema and the raw path identifier
        # Note: We don't resolve or fully validate the path here,
        #       that's left to the calling function (e.g., using os.path.isdir)
        return "local-dir", identifier

    # Check for Hugging Face Hub schema
    elif model_name.startswith(HF_HUB_PREFIX):
        # Extract the identifier (HF Hub ID) after the prefix
        identifier = model_name[len(HF_HUB_PREFIX) :]
        # Validate that the identifier is not empty
        if not identifier:
            raise ValueError("Empty identifier specified after 'hf-hub:' schema.")
        # Return the schema and the HF Hub ID
        return "hf-hub", identifier

    # If neither schema prefix is found
    else:
        # No schema detected, return None for schema and the original string as identifier
        return None, model_name


def _get_hf_config(
    model_id: str,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    revision: str | None = None,
):
    """Fetch model config from HuggingFace Hub."""
    config_path = download_pretrained_from_hf(
        model_id,
        filename="open_clip_config.json",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def get_model_config(model_name):
    """Fetch model config from schema specified location or local library configs."""
    loc, model_id = parse_model_name(model_name)
    if loc == "local-dir":
        local_path = Path(model_id) / "open_clip_config.json"
        with open(local_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model_cfg", config)
    elif loc == "hf-hub":
        config = _get_hf_config(model_id)
        return config.get("model_cfg", config)
    elif model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def load_state_dict(
    checkpoint_path: str,
    device="cpu",
    weights_only=True,
):
    # # Note: add_safe_globals never works, it needs exact same package versions as saved ckpt.
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        # NEW fix JIT models (a better fix is to update the model defs to use safetensors, which is
        # what open_clip did for openai pretrained weights).
        except RuntimeError as e:
            if "TorchScript" in str(e):
                checkpoint = torch.jit.load(checkpoint_path, map_location=device)
            else:
                raise RuntimeError(f"Error loading checkpoint {checkpoint_path} reraising") from e
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(
    model,
    checkpoint_path,
    strict=True,
    weights_only=True,
    resize_text_pos_emb: str = "cut",
):
    if Path(checkpoint_path).suffix in (".npz", ".npy"):
        # Separate path loading numpy big_vision (SigLIP) weights
        from .big_vision import load_big_vision_weights

        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path, weights_only=weights_only)
    # detect old format and make compatible with new format
    if "positional_embedding" in state_dict and not hasattr(model, "positional_embedding"):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    if "logit_bias" not in state_dict and model.logit_bias is not None:
        state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])
    # When loading a SigLIP model for non-SigLIP training set model.ckpt_loading_strict=False

    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = "text.transformer.embeddings.position_ids"
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]
    resize_pos_embed(state_dict, model)
    if resize_text_pos_emb not in ["", "none"]:
        resize_text_pos_embed(state_dict, model, interpolation=resize_text_pos_emb)

    # Initialize missing adapter weights
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if key not in state_dict:
            state_dict[key] = model_state_dict[key]

    miss_keys, unexp_keys = model.load_state_dict(state_dict, strict=False)
    if strict and len(unexp_keys) > 0:
        raise ValueError(f"Checkpoint mismatch {len(miss_keys)=} {len(unexp_keys)=}: {unexp_keys=}")
    elif len(unexp_keys) > 0:
        logger.warning(f"Checkpoint unexpected keys ({len(unexp_keys)}): {unexp_keys}")
    if len(miss_keys) > 0:
        logger.warning(
            f"Checkpoint missing keys ({len(miss_keys)}): {miss_keys[:5]} ... This can be expected "
            f"in some cases like when loading the text model from huggingface."
        )
    return miss_keys, unexp_keys


def load_checkpoint_without_text(model, checkpoint_path, weights_only=True):
    if Path(checkpoint_path).suffix in (".npz", ".npy"):
        raise NotImplementedError("big vision and load without text not implemented")

    state_dict = load_state_dict(checkpoint_path, weights_only=weights_only)
    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    if "logit_bias" not in state_dict and model.logit_bias is not None:
        state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])

    resize_pos_embed(state_dict, model)

    new_state_dict = {}
    for param_name, param in state_dict.items():
        # clip models where the text tower is in "text."
        if param_name.startswith("text."):
            continue
        # clip models where the text transformer is all over the place
        if param_name in {"positional_embedding", "text_projection"}:
            continue
        if any(param_name.startswith(p) for p in ["token_embedding.", "ln_final.", "transformer."]):
            continue

        new_state_dict[param_name] = param
    incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
    assert len(incompatible_keys.unexpected_keys) == 0, f"{incompatible_keys.unexpected_keys=}"

    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    force_preprocess_cfg: Optional[Dict[str, Any]] = None,
    pretrained_image: bool = False,  # Load default base image weights (at creation, if no CLIP weights)
    pretrained_text: bool = True,  # Load default base text weights (at creation, if no CLIP weights) - NEW
    cache_dir: Optional[str] = None,
    output_dict: bool = False,
    require_pretrained: bool = False,
    update_text_cfg_dict=None,
    hf_load_text_separately: bool = False,
    resize_text_pos_emb: str = "cut",
    weights_only: bool = True,
    # param init
    init_logit_scale: float = 2.659260036932778,  # np.log(1 / 0.07)
    init_logit_bias=None,
    model_loss_name: str = "clip",
    strict: bool = True,
    local_files_only: bool = False,
    revision: str | None = None,
    **model_kwargs,
):
    """
    Example model config:

    {
        "embed_dim": 512,
        "vision_cfg": {
            "timm_model_name": "vit_base_patch16_224",
            "timm_model_pretrained": False,
            "timm_pool": "",
            "timm_proj": "linear",
            "image_size": 224,
        },
        "text_cfg": {
            "hf_model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            "hf_tokenizer_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            "hf_proj_type": "mlp",
            "hf_pooler_type": "cls_last_hidden_state_pooler",
            "context_length": 256,
        },
    }
    """
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX) :]
        checkpoint_path = download_pretrained_from_hf(
            model_id, cache_dir=cache_dir, local_files_only=local_files_only, revision=revision
        )
        config = _get_hf_config(
            model_id, cache_dir, local_files_only=local_files_only, revision=revision
        )  # example: 'open_clip_config.json'.
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config["preprocess_cfg"])
        model_cfg = config["model_cfg"]
    else:
        model_name = model_name.replace("/", "-")  # for callers using old naming with / in ViT name
        checkpoint_path = None
        model_cfg = None
    if "pretrained_hf" in model_kwargs:
        # for backwards compat, override pretrained_text
        pretrained_text = model_kwargs.pop("pretrained_hf")
    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = model_cfg or get_model_config(model_name)
    if model_cfg is not None:
        logger.info(f"Loaded {model_name} model config.")
    else:
        logger.error(f"Model config for {model_name} not found; available models {list_models()}.")
        raise RuntimeError(f"Model config for {model_name} not found.")

    # modify model cfg
    if update_text_cfg_dict is not None:  # NEW text cfg dict updater
        for k, v in update_text_cfg_dict.items():
            model_cfg["text_cfg"][k] = v
    if force_quick_gelu:  # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True
    if force_patch_dropout is not None:  # override the default patch dropout value
        model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout
    if force_image_size is not None:  # override model config's image size
        model_cfg["vision_cfg"]["image_size"] = force_image_size
    is_timm_model = "timm_model_name" in model_cfg.get("vision_cfg", {})
    if pretrained_image:
        assert is_timm_model, "pretrained image towers currently only supported for timm models"
        model_cfg["vision_cfg"]["timm_model_pretrained"] = True

    # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
    cast_dtype = get_cast_dtype(precision)
    is_hf_model = "hf_model_name" in model_cfg.get("text_cfg", {})
    if is_hf_model:
        # load pretrained weights for HF text model IFF no CLIP weights being loaded
        model_cfg["text_cfg"]["hf_model_pretrained"] = pretrained_text and not pretrained
    custom_text = model_cfg.pop("custom_text", False) or force_custom_text or is_hf_model

    # TODO are these needed still?
    model_cfg["init_logit_scale"] = init_logit_scale
    model_cfg["init_logit_bias"] = init_logit_bias
    model_cfg["model_loss_name"] = model_loss_name

    if custom_text:
        if "multimodal_cfg" in model_cfg:
            model = CoCa(**model_cfg, cast_dtype=cast_dtype, output_dict=output_dict)
        else:
            model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype, output_dict=output_dict)
    else:
        model = CLIP(**model_cfg, cast_dtype=cast_dtype, output_dict=output_dict)

    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if "fp16" in precision else torch.bfloat16
        # manual mixed precision that matches original OpenAI behaviour
        if is_timm_model:
            # FIXME this is a bit janky, create timm based model in low-precision and
            # then cast only LayerNormFp32 instances back to float32 so they don't break.
            # Why? The convert_weights_to_lp fn only works with native models.
            model.to(device=device, dtype=dtype)
            from .transformer import LayerNormFp32

            def _convert_ln(m):
                if isinstance(m, LayerNormFp32):
                    m.weight.data = m.weight.data.to(torch.float32)
                    m.bias.data = m.bias.data.to(torch.float32)

            model.apply(_convert_ln)
        else:
            model.to(device=device)
            convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if "fp16" in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    pretrained_loaded = False
    if pretrained:
        checkpoint_path = ""
        pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        if pretrained_cfg:
            checkpoint_path = download_pretrained(
                pretrained_cfg,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )
            preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            pretrained_quick_gelu = pretrained_cfg.get("quick_gelu", False)
            model_quick_gelu = model_cfg.get("quick_gelu", False)
            if pretrained_quick_gelu and not model_quick_gelu:
                warnings.warn(
                    f"These pretrained weights were trained with QuickGELU activation but the "
                    f"model config does not have that enabled. Consider using a model config "
                    f"with a '-quickgelu' suffix or enable with a flag.\n"
                    f"model cfg: {model_cfg}\n"
                    f"pretrained cfg: {pretrained_cfg}\n"
                )
            elif not pretrained_quick_gelu and model_quick_gelu:
                warnings.warn(
                    f"The pretrained weights were not trained with QuickGELU but this activation "
                    f"is enabled in the model config, consider using a model config without "
                    f"QuickGELU or disable override flags.\n"
                    f"model cfg: {model_cfg}\n"
                    f"pretrained cfg: {pretrained_cfg}\n"
                )
        elif os.path.exists(pretrained):
            checkpoint_path = pretrained

        if checkpoint_path:
            logger.info(f"Loading pretrained {model_name} weights ({pretrained}).")
            load_checkpoint(
                model,
                checkpoint_path,
                resize_text_pos_emb=resize_text_pos_emb,
                weights_only=weights_only,
                strict=strict,
            )
        else:
            error_str = (
                f"Pretrained weights ({pretrained}) not found for model {model_name}."
                f" Available pretrained tags ({list_pretrained_tags_by_model(model_name)}."
            )
            logger.warning(error_str)
            raise RuntimeError(error_str)
        pretrained_loaded = True
    elif has_hf_hub_prefix:
        if hf_load_text_separately:  # NEW load HF text weights separately
            # we want to load visual and other params from the hf clip checkpoint
            # but text params from the hf text checkpoint.
            # in case text_cfg.hf_model_pretrained is True, those were already loaded.
            logger.info(f"Loading pretrained {model_name} text only weights ({checkpoint_path}).")
            _incompatible_keys = load_checkpoint_without_text(
                model, checkpoint_path, weights_only=weights_only
            )
            # model_cfg["text_cfg"]["hf_model_pretrained"] = True
            pretrained_loaded = model_cfg["text_cfg"]["hf_model_pretrained"]
        else:
            logger.info(f"Loading pretrained {model_name} weights ({checkpoint_path}).")
            load_checkpoint(
                model,
                checkpoint_path,
                resize_text_pos_emb=resize_text_pos_emb,
                weights_only=weights_only,
                strict=strict,
            )
            pretrained_loaded = True

    if require_pretrained and not pretrained_loaded:
        # callers of create_model_from_pretrained always expect pretrained weights
        raise RuntimeError(
            f"Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) "
            f"but not loaded."
        )

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, "image_size", None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg["size"] = model.visual.image_size
    model.model_cfg = model_cfg
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    # in some situations now the hf text encoder is set to eval mode, while the rest is train mode
    # the default state is all train mode, so set that here
    model.train()
    # for n, m in model.named_modules():
    #     str_ = "train" if m.training else "eval"
    #     cls = m.__class__.__name__
    #     print(f"  {str_}  {n} {cls}")
    return model


def get_tokenizer(
    model_name: str = "",
    context_length: Optional[int] = None,
    **kwargs,
):
    if model_name.startswith(HF_HUB_PREFIX):
        model_name = model_name[len(HF_HUB_PREFIX) :]
        try:
            config = _get_hf_config(model_name, cache_dir=kwargs.pop("cache_dir", None))[
                "model_cfg"
            ]
        except Exception as e:
            # # the tokenizer works so we can ignore this message
            # print(f"Error getting HF config for {model_name}: {type(e).__name__} {e}")
            tokenizer = HFTokenizer(
                model_name,
                context_length=context_length or DEFAULT_CONTEXT_LENGTH,
                **kwargs,
            )
            return tokenizer
    else:
        config = get_model_config(model_name)
        assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get("text_cfg", {})
    if "tokenizer_kwargs" in text_config:
        tokenizer_kwargs = dict(text_config["tokenizer_kwargs"], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get("context_length", DEFAULT_CONTEXT_LENGTH)

    if "hf_tokenizer_name" in text_config:
        tokenizer = HFTokenizer(
            text_config["hf_tokenizer_name"],
            context_length=context_length,
            **tokenizer_kwargs,
        )
    else:
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer


def create_loss(args):
    raise NotImplementedError("create_loss was removed, lit_open_clip.py manually creates the loss")


def create_model_and_transforms(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    image_interpolation: Optional[str] = None,
    image_resize_mode: Optional[str] = None,  # only effective for inference
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
    pretrained_image: bool = False,
    pretrained_hf: bool = True,
    cache_dir: Optional[str] = None,
    output_dict: Optional[bool] = None,
    update_text_cfg_dict=None,
    hf_load_text_separately: bool = False,
    resize_text_pos_emb: str = "cut",
    weights_only: bool = True,
    local_files_only: bool = False,
    revision: str | None = None,
    **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        update_text_cfg_dict=update_text_cfg_dict,
        hf_load_text_separately=hf_load_text_separately,
        resize_text_pos_emb=resize_text_pos_emb,
        weights_only=weights_only,
        local_files_only=local_files_only,
        revision=revision,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )
    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    image_interpolation: Optional[str] = None,
    image_resize_mode: Optional[str] = None,  # only effective for inference
    return_transform: bool = True,
    cache_dir: Optional[str] = None,
    update_text_cfg_dict=None,
    hf_load_text_separately: bool = False,
    resize_text_pos_emb: str = "cut",
    weights_only: bool = True,
    local_files_only: bool = False,
    **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
        require_pretrained=True,
        update_text_cfg_dict=update_text_cfg_dict,
        hf_load_text_separately=hf_load_text_separately,
        resize_text_pos_emb=resize_text_pos_emb,
        weights_only=weights_only,
        local_files_only=local_files_only,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess
