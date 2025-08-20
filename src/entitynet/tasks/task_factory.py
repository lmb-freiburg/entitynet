import torchvision
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError

from packg.misc import format_exception
from visiontext.configutils import load_dotlist

from entitynet.config.main_config import Config
from entitynet.config.task_config import BaseTaskCfg, ClipTaskC, find_task_file, load_task_config
from entitynet.datasets.dataset_factory import build_dataset_from_config
from entitynet.preprocessor.preprocessor_factory import build_vis_preprocessor_from_config
from entitynet.tasks.classification_task import ClipZeroshotClassificationTask
from entitynet.tasks.contrastive_task import ContrastiveRetrievalTask
from entitynet.tasks.contrastive_task_multitext import ContrastiveRetrievalTaskMultiText
from entitynet.tasks.sugarcrepepp_task import SugarCrepePPTask


def create_task_from_config(task_key: str, task_cfg: BaseTaskCfg):
    task_factory = task_cfg.task_factory
    if task_factory == ClipTaskC.ZS_CLS:
        return ClipZeroshotClassificationTask(task_key, task_cfg)
    if task_factory == ClipTaskC.CONTRASTIVE:
        return ContrastiveRetrievalTask(task_key, task_cfg)
    if task_factory == ClipTaskC.CONTRASTIVE_MULTITEXT:
        return ContrastiveRetrievalTaskMultiText(task_key, task_cfg)
    if task_factory == ClipTaskC.SUGARCREPEPP:
        return SugarCrepePPTask(task_key, task_cfg)

    raise ValueError(f"Unknown task name: {task_factory}")


def build_train_and_val_tasks(
    config: Config, eval_datasets_dict, eval_loader_dict, world_size: int = 1
):
    misconfig_str = ""
    if config.train_task is None:
        misconfig_str = f"{config.train_task=}"
    elif config.train_task.dataset is None:
        misconfig_str = f"{config.train_task.dataset=}"
    if misconfig_str != "":
        raise ValueError(
            f"Training not configured correctly: {misconfig_str} "
            f"Either fix config or pass --test_only"
        )

    # build train dataset
    train_vis_preproc = build_vis_preprocessor_from_config(config.train_task.vis_preproc)
    logger.info(f"Train vis preprocessor: {train_vis_preproc}")
    batch_size = config.trainer.batch_size
    dataset_cfg = config.train_task.dataset
    workers = config.trainer.workers
    train_dataset, train_dataloader = build_dataset_from_config(
        dataset_cfg,
        transform=train_vis_preproc,
        batch_size=batch_size,
        workers=workers,
        is_train=True,
        seed=config.trainer.seed,
        world_size=world_size,
    )
    # build train task
    train_task = create_task_from_config("train_task", config.train_task)

    # build validation tasks
    val_task_keys = config.trainer.val_task_keys
    val_loaders, val_task_cfgs, val_tasks, val_datasets = [], [], [], []
    for val_task_key in val_task_keys:
        if val_task_key not in config.eval_tasks:
            raise ValueError(
                f"Misconfiguration: requested validation task {val_task_key=} not found in "
                f"defined validation tasks {list(config.eval_tasks.keys())}"
            )
        val_task_cfg = config.eval_tasks[val_task_key]
        val_task_cfgs.append(val_task_cfg)
        val_task = create_task_from_config(val_task_key, val_task_cfg)
        val_tasks.append(val_task)
        val_dataset = eval_datasets_dict[val_task_key]
        val_datasets.append(val_dataset)
        val_loader = eval_loader_dict[val_task_key]
        val_loaders.append(val_loader)
        logger.info(
            f"Validation task {val_task_key} with dataset {val_task_cfg.dataset.dataset_name}/"
            f"{val_task_cfg.dataset.dataset_split} length {len(val_dataset)}"
        )
    return (
        train_task,
        train_dataset,
        train_dataloader,
        val_task_keys,
        val_task_cfgs,
        val_tasks,
        val_datasets,
        val_loaders,
    )


def build_test_tasks(config, eval_datasets_dict, eval_loader_dict):
    # build test tasks
    test_task_keys = config.trainer.test_task_keys
    if test_task_keys is None:
        raise ValueError(f"Misconfiguration: trainer.test_task_keys not defined.")
    if len(test_task_keys) == 0:
        raise ValueError(f"Misconfiguration: trainer.test_task_keys is length zero.")
    test_dict = {}
    for test_task_key in test_task_keys:
        test_task_cfg = config.eval_tasks[test_task_key]
        test_task = create_task_from_config(test_task_key, test_task_cfg)
        test_dataset = eval_datasets_dict[test_task_key]
        test_loader = eval_loader_dict[test_task_key]
        logger.info(
            f"Test task {test_task_key} with dataset {test_task_cfg.dataset.dataset_name}/"
            f"{test_task_cfg.dataset.dataset_split} length {len(test_dataset)}"
        )
        test_dict[test_task_key] = test_task_cfg, test_task, test_dataset, test_loader
    return test_dict


def get_dataset_and_task(
    task_name: str, options: list[str] | None = None, image_size=None, resize_mode=None
):
    """Helper to quickly build a task and dataset from a task name and options."""
    # create task config and build the task
    task_config_file = find_task_file(task_name)
    logger.info(f"Loading {task_config_file}")
    task_conf_omega: DictConfig = OmegaConf.load(task_config_file.as_posix())
    full_config = OmegaConf.create({"task": task_conf_omega})

    if options is not None:
        dict_dotlist = load_dotlist(options)
        logger.info(f"Enriching config with args.options: {dict_dotlist}")
        full_config = OmegaConf.merge(full_config, dict_dotlist)

    # some tasks try to load values from the model config, however there is no model here.
    # these kwargs allow setting these fields on the fly.
    if image_size is not None:
        dict_dotlist = load_dotlist([f"model.vis_preproc.clip_pp_cfg.size={image_size}"])
        full_config = OmegaConf.merge(full_config, dict_dotlist)
    if resize_mode is not None:
        dict_dotlist = load_dotlist([f"model.vis_preproc.clip_pp_cfg.resize_mode={resize_mode}"])
        full_config = OmegaConf.merge(full_config, dict_dotlist)
    try:
        full_config_dict = OmegaConf.to_container(full_config, resolve=True)
    except InterpolationKeyError as e:
        raise RuntimeError(
            f"Error resolving task config: {format_exception(e)}. When calling this function you "
            f"can override {image_size=} and {resize_mode=}. You can also pass a list of options "
            f"in format key1.key2=value to provide the missing keys."
        )
    task_conf_dict = full_config_dict["task"]

    # task_cfg: BaseTaskCfg = attrs_from_dict(BaseTaskCfg, task_conf_dict)
    task_cfg = load_task_config(task_conf_dict, task_name, task_config_file)
    logger.info(f"Resolved task config: {task_cfg}")
    task_key = task_name
    task = create_task_from_config(task_key, task_cfg)
    logger.info(f"Created task {task}")

    # but here there is no model
    if task_cfg.vis_preproc is not None:
        # if the task defines a preprocessing, use it
        transform = build_vis_preprocessor_from_config(task_cfg.vis_preproc)
    else:
        # otherwise we would use model preprocessing but there is no model loaded currently.
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    dataset_cfg = task_cfg.dataset
    logger.debug(f"Loading dataset for task {task_key}: {dataset_cfg}")
    batch_size = task_cfg.dataset.batch_size_eval
    dataset, _ = build_dataset_from_config(
        dataset_cfg,
        transform=transform,
        batch_size=batch_size,
        workers=0,
        is_train=False,
    )
    return dataset, task
