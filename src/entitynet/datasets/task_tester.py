from pprint import pprint

from omegaconf import DictConfig, OmegaConf

from packg.iotools.yamlext import load_yaml

from entitynet.config.model_config import PreprocCfg
from entitynet.config.task_config import load_task_config, resolve_tasks
from entitynet.datasets.dataset_factory import build_eval_dataset_for_task

_, _, available_tasks = resolve_tasks()


def run_task_tester(task_key):
    task_config_file = available_tasks[task_key]
    task_config_dict = load_yaml(task_config_file)
    pprint(task_config_dict)
    # some tasks need to know image size to create their custom preprocessor
    # so we need to merge those fields and resolve the config
    config_omegaconf = OmegaConf.create({"task": task_config_dict})
    dotlist = OmegaConf.from_dotlist(
        [
            "model.vis_preproc.clip_pp_cfg.size=224",
            "model.vis_preproc.clip_pp_cfg.resize_mode=shortest",
        ]
    )
    config_omegaconf: DictConfig = OmegaConf.merge(config_omegaconf, dotlist)
    task_config_dict = OmegaConf.to_container(config_omegaconf, resolve=True)["task"]
    # now everything is resolved and we can load the task config
    task_config = load_task_config(task_config_dict, task_key, task_config_file)

    print(task_config)

    # usually the preprocessor is given by the model, but here there is no model, so if there is no
    # task_cfg.vis_preproc set, we need to set it manually.
    # same goes if the task defines it as an unresolved $ omegaconf variable.
    if (
        task_config.vis_preproc is None
        or "$" in task_config.vis_preproc.preproc_ident
        or "$" in task_config.vis_preproc.preproc_factory
    ):
        vis_preproc_cfg = PreprocCfg(
            preproc_factory="open_clip",
            preproc_ident="ViT-B-32/laion2b_e16",
        )
        task_config.vis_preproc = vis_preproc_cfg
    dataset, loader = build_eval_dataset_for_task(task_key, task_config, workers=0, download=False)
    return dataset, loader
