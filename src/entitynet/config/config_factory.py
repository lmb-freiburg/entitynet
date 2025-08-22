from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from attr import asdict, fields_dict
from omegaconf import DictConfig, OmegaConf

from packg.iotools.jsonext import load_json
from packg.log import logger
from packg.typext import PathType
from typedparser import attrs_from_dict
from visiontext.configutils import load_dotlist

from entitynet.config.main_config import Config
from entitynet.config.model_config import BaseModelCfg, ModelConfigs
from entitynet.config.task_config import (
    BaseTaskCfg,
    ClipContrastiveTaskCfg,
    ClipTaskConfigs,
    find_task_file,
    load_task_config,
    resolve_task_lists,
)
from entitynet.models.clip_misc_utils import process_clip_model_name
from entitynet.paths import get_entitynet_output_dir, get_entitynet_repo_root
from entitynet.preprocessor.open_clip_prep import get_preprocess_config_file
from entitynet.preprocessor.preprocessor_factory import create_preprocessing_config_file_from_model


def load_config_from_file(
    config_file: PathType,
    merge_dotlist: list[str] | None = None,
    override_dict: dict | None = None,
) -> Config:
    """
    Load a config from a yaml file.
    Args:
        config_file: Path to the yaml file.
        merge_dotlist: List of dotlist strings to merge into the config.
            E.g. "trainer.batch_size=16", ... given by -o argument.
        override_dict: Nested dictionary of overrides to apply to the config.
    Returns:
        Config object.
    """
    logger.debug(f"Loading yaml into structure of crx/config.py - {config_file}")
    config_file = Path(config_file)
    conf: DictConfig = OmegaConf.load(config_file.as_posix())
    if merge_dotlist is not None:
        dict_dotlist = load_dotlist(merge_dotlist)
        conf = OmegaConf.merge(conf, dict_dotlist)
    conf_dict: dict = OmegaConf.to_container(conf, resolve=False)

    config = load_config_from_dict(conf_dict, override_dict=override_dict)

    # determine experiment name and output directory from the file name
    if config.trainer.experiment_name is None:
        cfg_path_rel = "configs/projects"
        if config_file.is_absolute():
            new_name = config_file.relative_to(get_entitynet_repo_root() / cfg_path_rel).as_posix()
        else:
            new_name = config_file.relative_to(cfg_path_rel).as_posix()
        new_name = ".".join(new_name.split(".")[:-1])  # remove .yaml
        config.trainer.experiment_name = new_name
    if config.trainer.output_dir is None:
        config.trainer.output_dir = (
            get_entitynet_output_dir() / "experiments" / config.trainer.experiment_name
        ).as_posix()
    elif not Path(config.trainer.output_dir).is_absolute():
        config.trainer.output_dir = (
            get_entitynet_output_dir() / config.trainer.output_dir
        ).as_posix()
    config.trainer.output_dir = Path(config.trainer.output_dir).as_posix()
    return config


def load_config_from_dict(config_dict: dict, override_dict: dict | None = None) -> "Config":
    """
    Internal function to load the main config from a resolved dictionary.
    Use Config.from_file() or Config.from_dict() instead.
    """
    # apply overrides to the config dict
    config_omegaconf = OmegaConf.create(config_dict)
    if override_dict is not None:
        config_omegaconf: DictConfig = OmegaConf.merge(config_omegaconf, override_dict)

    # The CLIP preprocessing is stored in JSON. That JSON is not loaded yet. Therefore fields of
    # model.vis_preproc.clip_pp_cfg are not filled yet and we cannot resolve the config to dict.
    # First, we need to resolve the openclip preprocessing config, and since that step also may
    # need to resolve something, we give it the omegaconf so it can lazily resolve if necessary.
    # Often the visual preprocessing is the same as the model preprocessing so it is referenced as
    # preproc_factory: ${model.model_factory} and preproc_ident: ${model.model_ident}
    config_omegaconf = _resolve_openclip_preproc(config_omegaconf)

    # Now all fields should be available and we can resolve the config to a dict.
    config_dict: dict = OmegaConf.to_container(config_omegaconf, resolve=True)

    # load sub configuration, and remove all special fields from the sub configs.
    # this way, config_dict will be compatible with Config class.
    config_dict, model_cfg = load_sub_config(
        config_dict, "model", "model_factory", ModelConfigs, BaseModelCfg
    )
    config_dict, train_task_cfg = load_sub_config(
        config_dict, "train_task", "task_factory", ClipTaskConfigs, BaseTaskCfg
    )
    # the eval tasks defined in external yaml files will be handled in postprocess_config.
    # however if they are defined in config_dict, we need to also load them as sub configs.
    eval_task_cfgs = {}
    for eval_task_key, eval_task_cfg in list(config_dict.get("eval_tasks", {}).items()):
        config_dict, eval_task_cfg = load_sub_config(
            config_dict,
            ("eval_tasks", eval_task_key),
            "task_factory",
            ClipTaskConfigs,
            BaseTaskCfg,
        )
        eval_task_cfgs[eval_task_key] = eval_task_cfg

    # load the simplified main config with only base fields for all sub configs
    try:
        config: Config = attrs_from_dict(Config, config_dict)
    except TypeError as e:
        raise TypeError(f"Failed to load config, see error above. Config:\n{config_dict}") from e

    # override with the real sub configs
    config.model = model_cfg
    config.train_task = train_task_cfg
    config.eval_tasks = eval_task_cfgs

    return postprocess_config(config)


def _resolve_openclip_preproc(config_omegaconf: DictConfig | dict) -> DictConfig:
    if not isinstance(config_omegaconf, DictConfig):
        # create a new omegaconf dict from the config dict
        config_omegaconf = OmegaConf.create(config_omegaconf)

    # resolve the model preprocessing config settings from open_clip
    # Note: it can actually be different for different pretrained weights.
    preproc_ident = config_omegaconf["model"]["vis_preproc"]["preproc_ident"]
    assert not preproc_ident.startswith("$"), (
        f"Unresolved OmegaConf node {preproc_ident} at model.vis_preproc.preproc_ident - "
        f"Either the config is incorrect or this config loader code is bugged."
    )
    preproc_model_name, preproc_pretrained = process_clip_model_name(preproc_ident)
    output_file = get_preprocess_config_file(preproc_model_name, preproc_pretrained)
    if not output_file.is_file():
        logger.warning(
            f"Preprocessor config file does not exist: {output_file} - Trying to create it. "
        )
        preproc_factory = config_omegaconf["model"]["vis_preproc"]["preproc_factory"]
        create_preprocessing_config_file_from_model(
            preproc_factory, preproc_model_name, preproc_pretrained
        )

    preproc_config = load_json(output_file)
    if "antialias" not in preproc_config:
        preproc_config["antialias"] = True

    # merge with the overwrites from the experiment config
    clip_pp_cfg = config_omegaconf["model"]["vis_preproc"].get("clip_pp_cfg")
    if clip_pp_cfg is not None:
        preproc_config.update(clip_pp_cfg)

    # write back to the config
    config_omegaconf["model"]["vis_preproc"]["clip_pp_cfg"] = preproc_config
    return config_omegaconf


def postprocess_config(config: Config):
    # if there is no vis_preproc set for training, use the one defined by the model
    if config.train_task is not None and config.train_task.vis_preproc is None:
        config.train_task.vis_preproc = deepcopy(config.model.vis_preproc)

    if config.eval_tasks is None:
        config.eval_tasks = {}
    if config.trainer.val_task_keys is None:
        config.trainer.val_task_keys = []
    if isinstance(config.trainer.val_task_keys, str):
        config.trainer.val_task_keys = [config.trainer.val_task_keys]
    if config.trainer.test_task_keys is None:
        config.trainer.test_task_keys = []
    if isinstance(config.trainer.test_task_keys, str):
        config.trainer.test_task_keys = [config.trainer.test_task_keys]

    # if any of the eval tasks refer to a task list e.g. "task_list::eval_list_objcls_imgn"
    # load the task list from yaml and convert to a list of task keys.
    config.trainer.val_task_keys = resolve_task_lists(config.trainer.val_task_keys)
    config.trainer.test_task_keys = resolve_task_lists(config.trainer.test_task_keys)

    # eval tasks can be either 1) defined directly in config.eval_tasks or 2) in yaml files.
    # here we load the yaml files and convert them to task configs.
    # we want to resolve nodes like ${model.model_factory} in those task configs.
    # but the tasks are separate yamls, so those nodes don't exist inside the task config.
    # so we need to create a copy of the main config, add the task config as a subnode,
    # resolve everything, and then get the task config back out.
    temp_config_base = DictConfig(asdict(config))
    all_eval_task_keys = sorted(set(config.trainer.val_task_keys + config.trainer.test_task_keys))
    for eval_task_key in all_eval_task_keys:
        if eval_task_key in config.eval_tasks:
            continue
        task_config_file = find_task_file(eval_task_key)
        task_conf_omega = OmegaConf.load(task_config_file.as_posix())
        temp_config = deepcopy(temp_config_base)
        temp_config["temp_task"] = task_conf_omega
        temp_config_dict = OmegaConf.to_container(temp_config, resolve=True)
        task_conf_dict = temp_config_dict["temp_task"]
        del temp_config, temp_config_dict
        task_conf_attrs = load_task_config(task_conf_dict, eval_task_key, task_config_file)
        config.eval_tasks[eval_task_key] = task_conf_attrs

    # postprocess tasks: store the key (name) of the task inside the task config, set default
    # batch_size_eval from trainer if not set, set default vis_preproc from model if not set
    for eval_key, eval_task in config.eval_tasks.items():
        eval_task.task_key = eval_key
        if eval_task.dataset.batch_size_eval is None:
            eval_task.dataset.batch_size_eval = deepcopy(config.trainer.batch_size_eval)
        if eval_task.vis_preproc is None:
            eval_task.vis_preproc = deepcopy(config.model.vis_preproc)

    return config


def verify_config(config: Config):
    for phase_name, keys in [
        ("val", config.trainer.val_task_keys),
        ("test", config.trainer.test_task_keys),
    ]:
        lossname2task = defaultdict(list)
        for key in keys:
            task_cfg = config.eval_tasks[key]
            if isinstance(task_cfg, ClipContrastiveTaskCfg):
                # it's a task that will log a loss
                loss_full_name = f"{phase_name}_loss{task_cfg.loss_name_appdx}"
                lossname2task[loss_full_name].append(key)
        lossname2task = dict(lossname2task)
        for loss_name, task_keys in lossname2task.items():
            if len(task_keys) > 1:
                raise ValueError(
                    f"Misconfig: Multiple tasks will log the same metric {loss_name} - tasks "
                    f"{task_keys}. Solution is to change the task config and either: "
                    f"1) set loss_name_appdx to non-empty string for all "
                    f"except one task, so loss will be logged under different names. "
                    f"2) Set disable_loss_logging=True for all but one task. "
                    f"3) Disable tasks until one is left."
                )


def load_sub_config(
    config_dict: dict,
    field_name: str | tuple[str, ...],
    factory_field_name: str,
    factory_mapping: dict[str, type],
    base_class: type,
) -> tuple[dict, Any]:
    """
    Load a configuration field with a different class based on the field's factory name.

    Args:
        config_dict: The full config dictionary
        field_name: Name of the field to load (e.g. "model") or tuple for nested fields.
        factory_field_name: Name of the field that contains the factory name (e.g. "model_factory")
        factory_mapping: Dictionary mapping factory names to their config classes
        base_class: Base class that all factory configs inherit from

    Returns:
        Tuple of (updated_config_dict, loaded_config_instance)
    """
    # extract and remove the field config from the main config dict
    if isinstance(field_name, str):
        field_cfg_dict = config_dict.pop(field_name)
    else:
        ref = config_dict
        for key in field_name[:-1]:
            ref = ref[key]
        field_cfg_dict = ref.pop(field_name[-1])
    if field_cfg_dict is None:
        return config_dict, None

    # initialize the config for the sub field
    factory_name = field_cfg_dict[factory_field_name]
    field_cfg_cls = factory_mapping[factory_name]
    field_cfg = attrs_from_dict(field_cfg_cls, field_cfg_dict)

    # create a simplified config dict, compatible with the base class
    base_fields = set(fields_dict(base_class).keys())
    field_cfg_base_dict = {k: v for k, v in field_cfg_dict.items() if k in base_fields}

    # add the simplified config dict to the main config dict
    if isinstance(field_name, str):
        config_dict[field_name] = field_cfg_base_dict
    else:
        ref = config_dict
        for key in field_name[:-1]:
            ref = ref[key]
        ref[field_name[-1]] = field_cfg_base_dict

    return config_dict, field_cfg
