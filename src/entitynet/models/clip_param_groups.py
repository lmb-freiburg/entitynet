from pprint import pprint

from loguru import logger
from torch import nn

from packg import format_exception

from entitynet.config.main_config import Config, OptimizerCfg
from entitynet.config.model_config import ClipModelCfg
from entitynet.loralib.utils import get_lora_parameters
from entitynet.models.clip_layer_num import get_layer_num_and_tower_name_for_clip


def get_clip_param_groups(model: nn.Module, config: Config):
    opt_cfg: OptimizerCfg = config.optimizer
    weight_decay: float = opt_cfg.hparams["weight_decay"]
    learning_rate: float = opt_cfg.hparams.get("lr")
    use_lora = False
    cm: ClipModelCfg = config.model
    if cm.lora_cfg is not None:
        # ----- lora specific parameter groups
        # TODO fix weight decay should be applied only to non-bias params of lora
        use_lora = cm.lora_cfg.use_lora

    if opt_cfg.vision_layer_decay_factor < 1 or opt_cfg.text_layer_decay_factor < 1:
        return get_clip_param_groups_with_layer_decay(model, opt_cfg)

    # ----- no lr decay specific parameter groups
    if use_lora:
        named_parameters = get_lora_parameters(model)
    else:
        named_parameters = list(model.named_parameters())
    param_groups_dict = {}
    for param_name, param in named_parameters:
        if not param.requires_grad:
            continue
        # get weight decay
        weight_decay_here = weight_decay
        group_name = "wd"
        if not check_requires_weight_decay(param_name, param):
            group_name = "nowd"
            weight_decay_here = 0.0

        # build the group if not exists
        if group_name not in param_groups_dict:
            param_groups_dict[group_name] = {
                "params": [],
                "param_names": [],
                "weight_decay": weight_decay_here,
            }
            if learning_rate is not None:
                param_groups_dict[group_name]["lr"] = learning_rate

        # add parameter to the group
        param_groups_dict[group_name]["params"].append(param)
        param_groups_dict[group_name]["param_names"].append(param_name)
    return param_groups_dict


def check_requires_weight_decay(param_name, param):
    """
    this function tries to guess for each possible clip model from openclip on how the
    weight decays should be set.

    see open_clip training main.py
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/main.py#L311
    bias, logit_scale, layer_norm and <2D params have no weight decay.
    everything else has weight decay including all projection layers, embeddings, convs etc.

    dino excludes only 1D and .bias params from weight decay
    https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L632

    karpathy nanogpt also only excludes <2D
    https://github.com/karpathy/nanoGPT/blob/master/model.py#L263

    timm only exclude <2D and bias by default
    https://github.com/pprp/timm/blob/master/timm/optim/optim_factory.py#L38

    """
    param_last_name = param_name.split(".")[-1].lower()
    # # open_clip disables weight decay for visual class_embedding since it is shape 1D
    # # (768,) visual.class_embedding, however text cls embedding has weight decay enabled.
    # # for now assuming this is a bug, and embeddings need weight decay.
    # if param.ndim < 2:  # 0D or 1D tensors never need weight decay
    #     return False
    if any(ln_name in param_name for ln_name in (".ln_", ".layernorm_", ".layer_norm_", ".norm.")):
        return False
    if any(bn_name in param_name for bn_name in (".bn_", ".batchnorm_", ".batch_norm_")):
        return False
    if any(gn_name in param_name for gn_name in (".gn_", ".groupnorm_", ".group_norm_")):
        return False
    if param_last_name in set(
        ("bias", "in_proj_bias", "logit_bias", "q_bias", "v_bias", "logit_scale")
    ):
        return False
    # sanity check whether we didn't miss any params
    if param_last_name in set(
        (
            "positional_embedding",
            "pos_embed",
            "text_projection",
            "proj",
            "class_embedding",
            "cls_token",
            "weight",
            "in_proj_weight",
            "w_lora_A",
            "w_lora_B",
        )
    ):
        return True
    logger.error(f"Not sure if this parameter requires weight_decay: {param_name}")
    if param.ndim < 2:  # 0D or 1D tensors never need weight decay
        return False
    return param.ndim > 1


def get_clip_param_groups_with_layer_decay(model: nn.Module, opt_cfg: OptimizerCfg):
    # ----- lr decay specific parameter groups
    vision_tower = model.visual
    vision_tower_name = type(vision_tower).__name__
    if vision_tower_name.lower() == "modifiedresnet":
        # to implement, figure out the resnet param to layer number map, e.g. from quicktune
        # github.com/machinelearningnuremberg/QuickTune/blob/main/timm/optim/optim_factory.py
        raise NotImplementedError("ResNet model not implemented for decay factor")

    try:
        # open_clip vision transformer
        num_max_vision_layer = len(model.visual.transformer.resblocks)
    except AttributeError:
        # timm vision transformer
        num_max_vision_layer = len(model.visual.trunk.blocks)

    # find the text transformer
    if hasattr(model, "text"):
        # customtextclip
        text_transformer = model.text.transformer
    else:
        # default clip
        text_transformer = model.transformer
    try:
        # default transformer
        num_max_text_layer = len(text_transformer.resblocks)
    except AttributeError:
        # custom text clip, e.g. bertmodel
        num_max_text_layer = len(text_transformer.encoder.layer)

    num_max_layer = max(num_max_vision_layer, num_max_text_layer)

    if use_lora:
        named_parameters = get_lora_parameters(model)
    else:
        named_parameters = list(model.named_parameters())
    names = [name for name, _ in named_parameters]
    try:
        layer_to_number_and_tower = {
            name: get_layer_num_and_tower_name_for_clip(
                name,
                num_max_vision_layer,
                num_max_text_layer,
                last_layer_same=opt_cfg.layer_decay_output_layer_has_highest_lr,
            )
            for name in names
        }
    except ValueError as e:
        # improved error message
        print(f"********** Failed finding layer numbers for model: {repr(type(model))}")
        print(model)
        pprint(names)
        print(f"********** Failed finding layer numbers for model: {repr(type(model))}")
        print(format_exception(e))
        # breakpoint()
        raise ValueError(f"Failed finding layer numbers for model: {repr(type(model))}") from e

    param_groups_dict = {}
    for param_name, param in named_parameters:
        if not param.requires_grad:
            continue
        # get layer number, tower name, group name
        layer_number, tower_name = layer_to_number_and_tower[param_name]
        if tower_name == "visual" or tower_name == "other":
            decay_factor = opt_cfg.vision_layer_decay_factor
            group_tower_name = "visual"
        elif tower_name == "text":
            decay_factor = opt_cfg.text_layer_decay_factor
            group_tower_name = "text"
        else:
            raise ValueError(f"Unknown tower name: {tower_name}")

        group_name_list = [f"layer{layer_number}"]
        if opt_cfg.vision_layer_decay_factor != opt_cfg.text_layer_decay_factor:
            group_name_list.append(group_tower_name)
        # get weight decay
        weight_decay_here = weight_decay
        if not check_requires_weight_decay(param_name, param):
            group_name_list.append("nowd")
            weight_decay_here = 0.0
        # get final group name
        group_name = "_".join(group_name_list)

        if opt_cfg.layer_decay_output_layer_has_highest_lr:
            # experiments we ran before: first layer is highest lr
            learning_rate_here = learning_rate * decay_factor ** (num_max_layer - layer_number + 1)
        else:
            # new experiments is with first layer lowest lr, same as quicktune
            learning_rate_here = learning_rate * decay_factor**layer_number

        # build the group
        if group_name not in param_groups_dict:
            param_groups_dict[group_name] = {
                "params": [],
                "param_names": [],
                "weight_decay": weight_decay_here,
                "lr": learning_rate_here,
            }
        else:
            assert param_groups_dict[group_name]["weight_decay"] == weight_decay_here
            assert param_groups_dict[group_name]["lr"] == learning_rate_here
        # add parameter to the group
        param_groups_dict[group_name]["params"].append(param)
        param_groups_dict[group_name]["param_names"].append(param_name)
    return param_groups_dict
