"""
Disentangles preprocessing from the model.

Note: Preprocessing can also be different for different pretrained weights.
"""

from pathlib import Path

from packg.iotools import load_json

from entitynet.models.clip_misc_utils import make_hf_model_name_safe
from entitynet.paths import get_entitynet_repo_root
from open_clip.transform import PreprocessCfg, image_transform_v2


def load_open_clip_preprocessor(
    model_name: str, pretrained: str, aug_cfg=None, is_train=False, **kwargs
):
    pp_cfg_dict = load_open_clip_preprocess_cfg(model_name, pretrained)
    # do not overwrite with None
    kwargs_no_none = {k: v for k, v in kwargs.items() if v is not None}
    pp_cfg_dict.update(kwargs_no_none)
    try:
        pp_cfg = PreprocessCfg(**pp_cfg_dict)
    except Exception as e:
        raise RuntimeError(f"{pp_cfg_dict=} {model_name=} {pretrained=}") from e
    return image_transform_v2(pp_cfg, is_train=is_train, aug_cfg=aug_cfg)


def load_open_clip_preprocessor_manually(aug_cfg=None, is_train=False, **kwargs):
    pp_cfg_dict = {k: v for k, v in kwargs.items() if v is not None}
    try:
        pp_cfg = PreprocessCfg(**pp_cfg_dict)
    except Exception as e:
        raise RuntimeError(f"{pp_cfg_dict=}") from e
    return image_transform_v2(pp_cfg, is_train=is_train, aug_cfg=aug_cfg)


def get_preprocess_config_file(model_name: str, pretrained: str) -> Path:
    output_dir = get_entitynet_repo_root() / f"src/open_clip/preprocess_configs"
    output_file = output_dir / f"{make_hf_model_name_safe(model_name)}__{pretrained}.json"
    return output_file


def load_open_clip_preprocess_cfg(model_name, pretrained):
    output_file = get_preprocess_config_file(model_name, pretrained)
    assert output_file.is_file(), f"Preprocess config file does not exist: {output_file}"
    return load_json(output_file)
