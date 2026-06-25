"""
EntityNet CLIP training and evaluation.
"""

import datetime
import os
import warnings
from pathlib import Path
from socket import gethostname
from timeit import default_timer

import lightning as lit
import torch
from attrs import asdict, define
from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from loguru import logger

from packg.debugging import connect_to_pycharm_debug_server
from packg.iotools import dump_yaml, dumps_yaml, yield_lines_from_file
from packg.log import get_logger_level_from_args
from packg.strings import format_pseudo_table
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from visiontext.distutils import WorldInfo, get_world_info

from entitynet.config.config_factory import load_config_from_file, verify_config
from entitynet.datasets.dataset_factory import build_eval_datasets
from entitynet.litext.ckpts import CustomModelCheckpoint
from entitynet.litext.cleanup_train_outputs import CleanupOutputsCallback
from entitynet.litext.evalloop import CustomEvalLoop
from entitynet.litext.neptuneext import rebuild_dict_no_nones
from entitynet.litext.progressbar import CustomTQDMProgressBar
from entitynet.models.model_factory import build_model_from_config
from entitynet.results.checkpoint_finder import find_checkpoints, find_ckpt_to_resume
from entitynet.tasks.task_factory import build_test_tasks, build_train_and_val_tasks
from entitynet.trainutils import figure_out_world_size, setup_loguru_train_logging

# disable this false positive warning: we accumulate metrics ourselves and only report from rank 0
warnings.filterwarnings(
    "ignore", message=r"It is recommended to use `self\.log\('val/.*, sync_dist=True.*"
)


@define
class Args(VerboseQuietArgs):
    config_file: Path = add_argument("config_file", type=str, help="Experiment config file")
    fast_dev_run: int = add_argument(type=int, help="Run only N batches", default=0)
    options: list[str] | None = add_argument(shortcut="-o", action="append", help="Override config")
    trace: str | None = add_argument(type=str, help="Connect debug server on this host.")
    trace_port: int = add_argument(type=int, default=33553, help="Target debugging server port")
    overwrite: bool = add_argument(
        action="store_true", help="Overwrite output if exists. Default is to skip the test."
    )
    test_only: bool = add_argument(action="store_true", help="Only run test, skip training")
    test_init: bool = add_argument(action="store_true", help="Test the initial checkpoint")
    test_last: bool = add_argument(
        action="store_true",
        help="Test the last checkpoint. Shortcut for setting trainer.test_last=True",
    )
    run_val: bool = add_argument(
        action="store_true", help="Validate the initial or loaded checkpoint and exit"
    )
    load_ckpt: str | None = add_argument(help="Load this ckpt file")
    run_id: str | None = add_argument(help="Run id for neptune logger and subfolder")
    with_id: str | None = add_argument(help="Resume run id for neptune logger")
    vislogger: str = add_argument(default="csv", help="Logger type: wandb, neptune")
    debug: bool = add_argument(action="store_true", help="Debug mode")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    if args.debug:
        args.vislogger = "csv"
        args.run_id = f"debug" if args.run_id is None else f"debug_{args.run_id}"
    # at this point lightning trainer doesn't exist so we get rank and world_size from the env
    global_rank, world_size = get_world_info()

    horst = gethostname()
    print(f"=============================== Init process on {horst} {global_rank=}, {world_size=}")
    if args.trace is not None and global_rank == 0:
        connect_to_pycharm_debug_server(args.trace, args.trace_port)

    # ----- load config and setup output dir
    config = load_config_from_file(args.config_file, merge_dotlist=args.options)
    verify_config(config)
    if args.test_last:
        config.trainer.test_last = True
    trcfg = config.trainer
    trcfg.output_dir = Path(trcfg.output_dir)
    if args.run_id is not None:
        trcfg.output_dir = trcfg.output_dir / args.run_id
    else:
        trcfg.output_dir = trcfg.output_dir / "defaultrun"
    if args.fast_dev_run:
        trcfg.output_dir = trcfg.output_dir.parent / f"fastdevrun-{trcfg.output_dir.name}"
    new_world_size = figure_out_world_size(config)

    # ----- setup console logger
    ckpt_dir = trcfg.output_dir / "ckpt"
    logger_level = get_logger_level_from_args(args)
    log_name = "test_only_log" if args.test_only else "train_log"
    setup_loguru_train_logging(trcfg.output_dir, logger_level, log_name)

    # ----- setup seed, print config
    logger.info(f"Hostname: {horst} Date {datetime.datetime.now()}")
    logger.info(f"{args}")
    logger.info(f"Experiment name: {config.trainer.experiment_name}")
    logger.info(f"Output dir: {trcfg.output_dir}")
    # setup_seeds(config.trainer.seed, deterministic=False)
    seed_everything(config.trainer.seed, workers=True)
    if config.trainer.print_config:
        # at this point we have too many eval tasks so do not print all of those
        config_as_dict = asdict(config)
        print_eval_tasks = config_as_dict.pop("eval_tasks", {})
        logger.info(f"---------- Config:\n{dumps_yaml(config_as_dict, standard_format=False)}")
        logger.info(f"---------- Eval tasks: {format_pseudo_table(list(print_eval_tasks.keys()))}")

    # ----- check existing output folder
    r_ckpt_file, r_epoch, r_step = find_ckpt_to_resume(
        trcfg.output_dir, behavior=config.trainer.on_exists
    )
    if 0 < config.trainer.max_steps <= r_step or 0 < config.trainer.max_epochs <= r_epoch:
        logger.info(
            f"Steps {r_step}/{config.trainer.max_steps} or "
            f"epochs {r_epoch}/{config.trainer.max_epochs} reached."
        )
        if not args.test_only:
            logger.info(f"Training is complete and argument --test_only is not set. Exiting.")
            return

    # save config to experiment output dir, do not overwrite training config when testing
    runconfig_yaml_file = trcfg.output_dir / "runconfig.yaml"
    if global_rank == 0 and not (args.test_only and runconfig_yaml_file.is_file()):
        dump_yaml(asdict(config), runconfig_yaml_file, create_parent=True)

    # build all evaluation datasets
    eval_datasets_dict, eval_loader_dict = build_eval_datasets(config)
    train_task, train_dataset, train_dataloader = None, None, None
    val_task_keys, val_task_cfgs, val_tasks, val_datasets, val_loaders = (None,) * 5
    if not args.test_only:
        (
            train_task,  # e.g. Constrastive3D: BaseTask, under which on_eval_end is defined
            train_dataset,
            train_dataloader,
            val_task_keys,
            val_task_cfgs,
            val_tasks,
            val_datasets,
            val_loaders,
        ) = build_train_and_val_tasks(config, eval_datasets_dict, eval_loader_dict, new_world_size)

    # ----- setup metric logger
    if args.vislogger == "neptune":
        from lightning.pytorch.loggers import NeptuneLogger
        from neptune.utils import stringify_unsupported

        logger.warning(
            f"***********************************************\n"
            f"neptune will crash your run if the internet connection breaks. use wandb instead.\n"
            f"***********************************************\n"
        )
        neptune_custom_run_id = args.run_id
        neptune_with_id = args.with_id
        if neptune_with_id is not None:
            logger.warning(f"Ignoring neptune custom run_id, because with_id is used for resuming.")
            neptune_custom_run_id = None
        wlogger = NeptuneLogger(
            log_model_checkpoints=False,
            project=config.trainer.project_name,
            name=config.trainer.experiment_name,
            with_id=neptune_with_id,
            custom_run_id=neptune_custom_run_id,
            source_files=[],
            git_ref=False,
            capture_stdout=False,
            capture_stderr=False,
        )
        warnings.filterwarnings("ignore", message=".*It is recommended to use.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="neptune.new")
        log_config = rebuild_dict_no_nones(asdict(config))
        log_config.pop("eval_tasks", None)
        # if len(log_config["eval_tasks"]) > 10:
        #     logger.warning("Only logging config of first 10 eval tasks to neptune.")
        #     log_config["eval_tasks"] = dict(list(log_config["eval_tasks"].items())[:10])
        wlogger.experiment["config"] = stringify_unsupported(log_config)
        logger.info(f"Connected Neptune: {wlogger}")
        # write neptune id to file
        neptune_id = str(wlogger.version)
        neptune_id_file = Path(trcfg.output_dir) / "neptune_id.txt"
        if neptune_id_file.is_file():
            existing_ids = list(yield_lines_from_file(neptune_id_file))
        else:
            existing_ids = []
        if (
            neptune_id not in existing_ids
            and neptune_id is not None
            and str(neptune_id).lower() != "none"
        ):
            with open(neptune_id_file, "a") as f:
                f.write(neptune_id + "\n")
    elif args.vislogger == "csv":
        wlogger = CSVLogger(config.trainer.output_dir)
    elif args.vislogger == "wandb":
        import wandb

        project = os.environ.get("WANDB_PROJECT", config.trainer.project_name)
        if project is None:
            raise ValueError(
                "Either set WANDB_PROJECT environment variable or set config.trainer.project_name"
            )
        if not wandb.api.api_key:
            logger.warning("wandb API key not found. Attempting to login...")
            wandb.login()
        wlogger = WandbLogger(
            save_dir=config.trainer.output_dir / "wandb",
            project=project,
            name=args.run_id,
            version=args.run_id,
            id=args.run_id,
            resume="allow",
            # settings=wandb.Settings(...),
        )
    else:
        raise ValueError(f"Unknown vislogger: {args.vislogger}")

    # ----- create callbacks
    callbacks = []
    if not args.test_only:
        # ----- checkpoint callback
        ckpt_cfg = config.trainer.ckpt  # e.g. [verbose: true, monitor: "val_loss"]
        monitor = ckpt_cfg.monitor
        filename_appdx = ""
        if monitor is not None:
            monitor_str = monitor.replace("/", "_")
            filename_appdx = f"-{ckpt_cfg.mode}-{monitor_str}-" + "{" + monitor + ":.6f}"
        checkpoint_callback = CustomModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch}-{step}" + filename_appdx,
            monitor=ckpt_cfg.monitor,
            verbose=ckpt_cfg.verbose,
            save_last=True,
            save_top_k=ckpt_cfg.save_top_k,
            mode=ckpt_cfg.mode,
            auto_insert_metric_name=False,
            save_weights_only=False,
            every_n_train_steps=ckpt_cfg.every_n_train_steps,
            every_n_epochs=ckpt_cfg.every_n_epochs,
            enable_version_counter=False,
        )
        checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"  # type: ignore
        callbacks.append(checkpoint_callback)
    callbacks += [
        CustomTQDMProgressBar(refresh_rate=50),
        # DeviceStatsMonitor(),  # outputs way too much things to the console and logger
        CleanupOutputsCallback(trcfg.output_dir, keep_output_mode=trcfg.keep_output_mode),
    ]

    # ---------- create model
    model = build_model_from_config(config)
    logger.debug(model)
    if global_rank == 0:
        # build average of weights
        t1 = default_timer()
        param_list = []
        for param_key, param_value in model.named_parameters():
            param_list.append(param_value.reshape(-1))
        if len(param_list) == 0:
            logger.warning(f"Model with 0 parameters! Dummy model?")
        else:
            ps = torch.cat(param_list)
            logger.info(
                f"All params: n={ps.shape[0]:,d} mean={ps.mean().item():.3e} std={ps.std().item():.3e} "
                f"in {default_timer()-t1:.3f}s"
            )
    if not args.test_only:
        # configure one train and multiple validation tasks
        model.setup_train_task(train_task, train_dataset)  # assign args to model self
        model.setup_validation_tasks(val_tasks, val_datasets)  # assign args to model self

    # ----- create trainer
    if config.trainer.set_float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(config.trainer.set_float32_matmul_precision)
    if args.fast_dev_run > 0:
        # make sure validation runs after the fast_dev_run is done
        config.trainer.val_check_interval = args.fast_dev_run

    # note: to disable graceful shutdown on errors, edit trainer/call.py _call_and_handle_interrupt
    trainer = lit.Trainer(
        accelerator=config.trainer.accelerator,
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
        strategy=config.trainer.strategy,
        # strategy="ddp_find_unused_parameters_true", #TODO: dele if unused parameter issue fixed
        precision=config.trainer.precision,
        default_root_dir=config.trainer.output_dir,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        logger=wlogger,
        max_epochs=config.trainer.max_epochs,
        max_steps=config.trainer.max_steps,
        val_check_interval=config.trainer.val_check_interval,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        inference_mode=True,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        log_every_n_steps=config.trainer.log_every_n_steps,
    )
    world_info = WorldInfo(trainer)

    # ----- lightning cannot handle the way the metrics are logged in our code well.
    # so use custom validation and test loop
    trainer.validate_loop = CustomEvalLoop(
        trainer, TrainerFn.VALIDATING, RunningStage.VALIDATING, inference_mode=True
    )
    trainer.test_loop = CustomEvalLoop(
        trainer, TrainerFn.TESTING, RunningStage.TESTING, inference_mode=True
    )

    # ----- run epoch 0 validation and exit
    if args.run_val:
        trainer.validate(model, val_loaders, ckpt_path=r_ckpt_file)
        logger.info(f"Validation done, exiting.")
        return

    # ----- fit
    if not args.test_only:
        # we cannot run trainer.validate because it doesn't properly setup distributed training
        world_info.print_with_rank(f"Call trainer.fit")
        trainer.fit(model, train_dataloader, val_loaders, ckpt_path=r_ckpt_file)
        world_info.print_with_rank(f"Done with trainer.fit")

    # ----- prepare test checkpoints
    ckpts_to_test = []
    last_ckpt, best_ckpt, _ = find_checkpoints(ckpt_dir)
    logger.info(f"Found {last_ckpt=} {best_ckpt=}")
    if last_ckpt is None:
        logger.warning(f"trainer.fit was called, but no last checkpoint exists in {ckpt_dir}")
    if config.trainer.test_last and last_ckpt is not None:
        ckpts_to_test.append(last_ckpt)
    if config.trainer.test_best:
        if best_ckpt is None:
            logger.error(f"test_best=True but no best checkpoint found in {ckpt_dir}")
        else:
            ckpts_to_test.append(best_ckpt)
    if args.load_ckpt is not None:
        logger.info(f"Loading custom checkpoint {args.load_ckpt}")
        ckpts_to_test.append({"file": args.load_ckpt, "epoch": 0, "global_step": 0})
    if args.test_init or (args.test_only and len(ckpts_to_test) == 0):
        ckpts_to_test.append({"file": None, "epoch": 0, "global_step": 0})
    if len(ckpts_to_test) == 0:
        raise ValueError(
            f"No checkpoints to test found in {ckpt_dir}. Pass --test_init to test "
            f"the model without training it."
        )

    # build test tasks
    test_task_keys = config.trainer.test_task_keys
    if test_task_keys is None or len(test_task_keys) == 0:
        logger.info(f"Nothing to test, exiting. {test_task_keys=}")
        return
    test_dict = build_test_tasks(config, eval_datasets_dict, eval_loader_dict)
    test_task_keys = list(test_dict.keys())

    # loop over all test tasks and run them
    for ckpt_to_test in ckpts_to_test:
        if ckpt_to_test is None:
            raise ValueError(
                "Checkpoint is None. Probably the training crashed and there are no checkpoints:\n"
                f"{ckpt_dir}"
            )
        r_ckpt_file = ckpt_to_test["file"]
        if r_ckpt_file is not None:
            logger.info(f"Loading new weights before testing: {r_ckpt_file}")
        else:
            logger.info(f"Testing with keeping the existing weights (trained or pretrained)")

        # during testing phase the epoch identifier must be set manually
        model.epoch_identifier = f"{ckpt_to_test['epoch']}-{ckpt_to_test['global_step']}"
        model.eval_phase = "test"

        test_tasks, test_datasets, test_loaders = [], [], []
        for test_task_key, (
            test_task_cfg,
            test_task,
            test_dataset,
            test_loader,
        ) in test_dict.items():
            output_file = model.get_eval_output_file(test_task_key, suffix="results.json")
            if output_file.is_file():
                logger.warning(f"Skip test because output exists: {output_file.as_posix()}")
                continue
            test_tasks.append(test_task)
            test_datasets.append(test_dataset)
            test_loaders.append(test_loader)
        if len(test_tasks) == 0:
            logger.warning(f"No tasks to test for {ckpt_to_test}")
            continue

        model.setup_test_tasks(test_tasks, test_datasets)
        world_info.print_with_rank(f"Testing!")
        metrics = trainer.test(model, test_loaders, ckpt_path=r_ckpt_file)
        if world_info.is_global_zero:
            # output_file = output_dir / f"ckpt-{model.epoch_identifier}_task-{task_key}_metrics.json"
            # dump_json(metrics, output_file)  # base_model is already saving those so no need
            if len(metrics) == 0:
                logger.error(f"No metrics returned after testing with {test_task_keys=}")
            else:
                logger.info(f"Metrics returned from test: {metrics}")
    logger.info(f"Results are in {trcfg.output_dir}")


def profile_stuff(model):
    # === BEGIN profiling stuff ===
    print()
    print()
    print()
    print("FlopCountAnalysis in newtrain.py")
    from fvcore.nn import FlopCountAnalysis

    if model.config.model.num_textmix_tokens == 0:
        dummy_batch = {
            "image": torch.randn(10, 3, 224, 224),
            "text": "some description of the image",
            "label": torch.tensor([1 for _ in range(10)]),
        }
    else:
        n_txt = model.config.model.num_textmix_tokens
        dummy_batch = {
            "image": torch.randn(10, 3, 224, 224),
            "text_list": ["some description of the image" for _ in range(n_txt)],
            "label": torch.tensor([1 for _ in range(10)]),
        }
    batch = model._prepare_train_batch(dummy_batch)
    fc_model = FlopCountAnalysis(model.model, (batch["image"], batch["tokens"]))
    print(" ### FVCORE total:", fc_model.total())
    from torchinfo import summary

    img = torch.randn((1, 3, 224, 224))
    tok32 = torch.ones((3, 32, model.model.token_embedding.embedding_dim))
    tok49 = torch.ones((3, 49, model.model.token_embedding.embedding_dim))
    tok64 = torch.ones((3, 64, model.model.token_embedding.embedding_dim))
    tok77 = torch.ones((3, 77, model.model.token_embedding.embedding_dim))
    print(
        "transformer 32\n:",
        summary(
            model.model.transformer,
            input_data=(tok32,),
            depth=0,
            col_names=["num_params", "mult_adds"],
        ),
    )
    print()
    print(
        "transformer 49\n:",
        summary(
            model.model.transformer,
            input_data=(tok49,),
            depth=0,
            col_names=["num_params", "mult_adds"],
        ),
    )
    print()
    print(
        "transformer 64\n:",
        summary(
            model.model.transformer,
            input_data=(tok64,),
            depth=0,
            col_names=["num_params", "mult_adds"],
        ),
    )
    print()
    print(
        "transformer 77\n:",
        summary(
            model.model.transformer,
            input_data=(tok77,),
            depth=0,
            col_names=["num_params", "mult_adds"],
        ),
    )
    print()
    print(
        "visual:       \n:",
        summary(
            model.model.visual, input_data=(img), depth=0, col_names=["num_params", "mult_adds"]
        ),
    )
    print()
    print()
    print()
    print()
    print(
        "transformer 32\n:",
        summary(
            model.model.transformer,
            input_data=(tok32,),
            depth=30,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
        ),
    )
    breakpoint()
    print()
    print()
    print()
    # === END profiling stuff ===


if __name__ == "__main__":
    main()
