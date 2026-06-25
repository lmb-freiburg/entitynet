"""
Note that the expression var: str = None means the variable is mandatory to set in the yaml,
otherwise the parser will raise an error.
"""

from __future__ import annotations

from typing import Any, Optional

from attr import define

from packg import Const


@define(auto_attribs=True, kw_only=True)
class BaseModelCfg:
    model_factory: str = None
    model_ident: str = None
    vis_preproc: PreprocCfg = None
    # during eval, there may not be a loss. but we may need the loss name to find out specifics
    # about the model forward pass, therefore store it at model level instead of training task level
    model_loss_name: str = "clip"
    ckpt_loading_strict: bool = True  # whether to use strict loading for model checkpoints


@define(auto_attribs=True, kw_only=True)
class PreprocCfg:
    preproc_factory: str = "${model.model_factory}"
    preproc_ident: str = "${model.model_ident}"
    aug_cfg: dict[str, Any] | None = None
    clip_pp_cfg: ClipPreprocCfg | None = None


# usually these will be determined by the model name and pretrained weights, but can be overwritten
@define(auto_attribs=True, kw_only=True)
class ClipPreprocCfg:
    size: int | tuple[int, int] | None = None
    mode: str | None = None
    mean: tuple[float, ...] | float | None = None
    std: tuple[float, ...] | float | None = None
    interpolation: str | None = None
    resize_mode: str | None = None  # shortest, squash, longest, none, see get_resize_transforms()
    fill_color: int | None = None
    antialias: bool | None = None


@define(auto_attribs=True, kw_only=True)
class DummyClipModelCfg(BaseModelCfg):
    attr_mode: str = "random_uniform"  # random_uniform, fixed_zero, fixed_one


@define(auto_attribs=True, kw_only=True)
class ClipLoraCfg:
    use_lora: bool | None = False
    params: Optional[list[str]] | None = ["q", "k", "v", "o"]
    encoder: str | None = None
    backbone: str | None = "ViT-B-32"
    position: str | None = "all"
    r: int | None = 2
    alpha: float | None = 1
    bias: str | None = None
    dropout_rate: float | None = 0.25


@define(auto_attribs=True, kw_only=True)
class ClipModelCfg(BaseModelCfg):
    tokenizer_name: str | None = None  # manually set tokenizer name, otherwise will use model name
    hf_text_encoder_name: str | None = None  # force custom hf text encoder, independent of model
    context_length: int = 32
    resize_text_pos_emb: str = "cut"  # cut, none, linear, bilinear etc.
    force_patch_dropout: float = 0.0
    force_custom_text: bool = False
    weights_only: bool = True

    # locking one of the towers
    lock_image_encoder: bool = False
    lock_text_encoder: bool = False  # Lock full text tower by disabling gradients
    # note: lock_text_encoder needs force_custom_text: true to work properly.
    # Leave last n image tower layer groups unlocked
    lock_image_unlocked_groups: int = 0
    lock_image_freeze_bn_stats: bool = (
        False  # Freeze BatchNorm running stats in image tower for any locked layers
    )
    # Leave last n text tower layer groups unlocked
    lock_text_unlocked_layers: int = 0
    lock_text_freeze_layer_norm: bool = (
        False  # Freeze LayerNorm running stats in text tower for any locked layers
    )
    lora_cfg: ClipLoraCfg | None = None


class ModelFactoryC(Const):
    NONE = "none"
    OPEN_CLIP = "open_clip"
    OPEN_CLIP_MANUAL = "open_clip_manual"  # do not use model_name to instantiate preprocessing


# for now preprocessor and model factories match
PreprocessorFactoryC = ModelFactoryC

ModelConfigs = {
    ModelFactoryC.OPEN_CLIP: ClipModelCfg,
}
