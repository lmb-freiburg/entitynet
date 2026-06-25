from __future__ import annotations

from pathlib import Path
from typing import Any

from attrs import define
from natsort import natsorted

from packg import Const, format_exception
from packg.iotools import load_yaml
from typedparser import attrs_from_dict

from entitynet.config.model_config import PreprocCfg
from entitynet.paths import get_entitynet_repo_root

# ---------- Definition of task configs ----------


@define(auto_attribs=True, kw_only=True)
class BaseTaskCfg:
    task_key: str | None = None  # will be set by post_init
    task_factory: str = None
    dataset: DatasetCfg | None = None
    vis_preproc: PreprocCfg | None = None  # in case the task needs different preproc than the model


@define(auto_attribs=True, kw_only=True)
class ClipZsClsTaskCfg(BaseTaskCfg):
    clip_zs_template: str = None
    use_synonyms: bool = False


@define(auto_attribs=True, kw_only=True)
class ClipContrastiveTaskCfg(BaseTaskCfg):
    # for the contrastive loss train task
    loss_name: str | None = None
    loss_local: bool = False
    loss_gather_with_grad: bool = False
    run_retrieval: bool = True  # set False for loss only task to save memory


@define(auto_attribs=True, kw_only=True)
class ClipContrastiveMultiTextCfg(BaseTaskCfg):
    n_texts_per_image: int = None


# ---------- Definition of task types and their mapping to config classes ----------


class ClipTaskC(Const):
    ZS_CLS = "zeroshot_classification"
    CONTRASTIVE = "contrastive"
    CONTRASTIVE_MULTITEXT = "contrastive_multitext"  # 1 image <-> N texts
    SUGARCREPEPP = "sugarcrepepp"


ClipTaskConfigs = {
    ClipTaskC.ZS_CLS: ClipZsClsTaskCfg,
    ClipTaskC.CONTRASTIVE: ClipContrastiveTaskCfg,
    ClipTaskC.CONTRASTIVE_MULTITEXT: ClipContrastiveMultiTextCfg,
    ClipTaskC.SUGARCREPEPP: BaseTaskCfg,
}


def load_task_config(
    task_conf_dict: dict,
    eval_task_key_for_error_msg: str | None = None,
    task_config_file_for_error_msg: Path | None = None,
) -> BaseTaskCfg:
    """
    Load a task config from a dictionary and map it to the appropriate task class.
    """
    task_factory = task_conf_dict["task_factory"]
    factory_mapping = ClipTaskConfigs
    task_class = factory_mapping[task_factory]
    try:
        task_conf_attrs = attrs_from_dict(task_class, task_conf_dict)
    except Exception as e:
        raise ValueError(
            f"Failed to load task config for '{eval_task_key_for_error_msg}' from file: "
            f"{task_config_file_for_error_msg} dict {task_conf_dict}. Error: {format_exception(e)}"
        ) from e
    return task_conf_attrs


# ---------- dataset config ----------


@define(auto_attribs=True, kw_only=True)
class DatasetCfg:
    dataset_factory: str = None
    dataset_name: str = None
    dataset_split: str = None
    batch_size_eval: int | None = None
    max_datapoints: int | None = None
    max_shards: int | None = None  # webdataset only
    eval_type: str = "default"
    deterministic_seed: int | None = None
    # TODO consider creating separate configs for separate dataset types
    # entitynet only
    text_aug: EntityNetTextAugCfg | None = None
    filter_dict: dict[str, Any] | None = None
    filter_op: str = "any"  # skip if "any" query matches or skip if "all" queries match.
    # meta merge multiple datasets
    merge_datasets: dict[str, DatasetCfg] | None = None
    merge_transforms: dict[str, PreprocCfg] | None = None


class EntityNetTextReturnMode(Const):
    SAMPLE = "sample"
    ALL = "all"
    SAMPLE_RINCE = "sample_rince"


@define(auto_attribs=True, kw_only=True)
class EntityNetTextAugCfg:
    n_texts_per_image: int = 0  # >0 = returns list of str, 0 = single str
    return_mode: str = EntityNetTextReturnMode.SAMPLE
    replace_noun_synonym_chance: float = 0.0
    replace_noun_definition_chance: float = 0.0
    replace_noun_hierarchy_chance: float = 0.0
    replace_noun_hierarchy_chance_living: float = 0.0  # the naturalentity is worse. less replace.
    replace_attr_query: float = 0.0  # replace the original search query with a new one
    alt_text_chance: float = 0.0  # how often to use alt-text vs query
    attronly_keep_query: float = 1.0
    attronly_replace_query_with_synonym: float = 0.0
    attronly_build_pseudo_query: float = 0.0
    attronly_strgf_replace_entity: float = 0.0
    attronly_attribute_only: float = 0.0
    attronly_replace_with_definition: float = 0.0
    attrnoun_keep_query: float = 1.0
    attrnoun_replace_query_with_synonym: float = 0.0
    attrnoun_build_pseudo_query: float = 0.0
    attrnoun_strgf_replace_entity: float = 0.0
    attrnoun_attribute_only: float = 0.0
    attrnoun_replace_with_definition: float = 0.0
    combine_synonym_and_parentsynonym: float = 0.0  # does not look so good for now.
    clip_prompts: float = 0.0  # use clip prompts for nouns


# ---------- Utility functions for task yamls and task list yamls ----------


def resolve_tasks():
    """
    Build a list of all available task yaml files.

    We want to be able to use arbitary subdirectories in the task config directory, so we
    resolve all yamls in that directory, and store the result globally.
    """
    global AVAILABLE_TASKS
    glob_str = "**/*.yaml"
    task_dirs = [get_entitynet_repo_root() / "configs/tasks"]
    if AVAILABLE_TASKS is None:
        AVAILABLE_TASKS = {}
        for task_dir in task_dirs:
            task_yaml_files = natsorted(task_dir.glob(glob_str))
            for f in task_yaml_files:
                key = f.stem
                if key in AVAILABLE_TASKS:
                    raise ValueError(
                        f"Task key '{key}' defined in multiple files: "
                        f"{AVAILABLE_TASKS[key]} and {f}"
                    )
                AVAILABLE_TASKS[key] = f
        # print(f"Resolved {len(AVAILABLE_TASKS)} tasks")
    return task_dirs, glob_str, AVAILABLE_TASKS


def find_task_file(eval_task_key: str) -> Path:
    """Find a task config file for the given task key."""
    task_dirs, glob_str, available_tasks = resolve_tasks()
    if eval_task_key not in available_tasks:
        raise ValueError(
            f"Evaluation task requested: '{eval_task_key}' but it was not defined in "
            f"the main config, and also doesn't exist as file in the task directories: "
            f"{task_dirs} searching for {glob_str}"
        )
    task_config_file = available_tasks[eval_task_key]
    return task_config_file


AVAILABLE_TASKS: dict[str, Path] | None = None


TASK_LIST_PREFIX = "task_list::"


def resolve_task_lists(task_keys: list[str]) -> list[str]:
    """
    Handling of task lists.

    A task list is refered in the config as a task like "task_list::eval_list_objcls_imgn".
    This loads the file task_lists/.../eval_list_objcls_imgn.yaml and adds all task_keys in the
    list to the config.

    Args:
        task_keys: List of task keys from the config. All task lists will be resolved.

    Returns:
        List of task keys with all task lists resolved.
    """
    new_val_task_keys = []
    for task_key in task_keys:
        if task_key.startswith(TASK_LIST_PREFIX):
            task_list_name = task_key[len(TASK_LIST_PREFIX) :]
            task_list_file = get_entitynet_repo_root() / f"configs/task_lists/{task_list_name}.yaml"
            if not task_list_file.is_file():
                raise FileNotFoundError(f"Task list file not found: {task_list_file}")
            task_list_data = load_yaml(task_list_file)["task_keys"]
            assert isinstance(task_list_data, list), f"Invalid {type(task_list_data)=}"
            assert len(task_list_data) > 0, f"Empty task list: {task_list_data=}"
            for taski, task in enumerate(task_list_data):
                assert isinstance(task, str), f"Invalid {task=} pos {taski} in {task_list_data=}"
            new_val_task_keys.extend(task_list_data)
        else:
            new_val_task_keys.append(task_key)
    new_val_task_keys = list({task_key: None for task_key in new_val_task_keys}.keys())
    return new_val_task_keys
