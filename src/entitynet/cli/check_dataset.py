"""
Load a dataset and print some samples.

Either by
- specifiing the config via a list of -o options
- or by specifying a config file, optional -o overrides, and --train_set or --val_set VAL_SET_NAME
"""

from attrs import define
from loguru import logger

from entitynet.config.config_factory import load_config_from_file
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument, attrs_from_dict
from visiontext.configutils import load_dotlist

from entitynet.config.task_config import DatasetCfg
from entitynet.datasets.dataset_factory import build_dataset_from_config
from entitynet.preprocessor.preprocessor_factory import get_simple_transform_to_tensor


@define
class Args(VerboseQuietArgs):
    config_file: str | None = add_argument(shortcut="-c", type=str, help="Path to config file")
    options: list[str] | None = add_argument(shortcut="-o", action="append", help="Override config")
    val_set: str | None = add_argument(type=str, help="Check validation set with this name")
    n_show: int = add_argument(type=int, default=10, help="Number of samples to show")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    test_deterministic = True
    if args.config_file is None:
        if args.options is None:
            raise ValueError("Either --config_file or --options must be specified")
        else:
            dict_dotlist = load_dotlist(args.options)
            logger.debug(f"dict_dotlist: {dict_dotlist}")
            dataset_cfg = attrs_from_dict(DatasetCfg, dict_dotlist)
    else:
        config = load_config_from_file(args.config_file, merge_dotlist=args.options)
        if args.val_set is not None:
            if args.val_set not in config.eval_tasks:
                # parse the config again and this time try adding the requested val_set
                options = args.options if args.options is not None else []
                options += [f"trainer.test_task_keys={args.val_set}"]
                config = load_config_from_file(args.config_file, merge_dotlist=options)
            dataset_cfg = config.eval_tasks[args.val_set].dataset
        else:
            logger.info(f"--- No --val_set specified, using train set")
            dataset_cfg = config.train_task.dataset

    logger.info(f"dataset_cfg: {dataset_cfg}")
    transform = get_simple_transform_to_tensor(224, mean=0, std=1)
    dataset, loader = build_dataset_from_config(
        dataset_cfg, batch_size=1, transform=transform, workers=0, is_train=False
    )
    print(f"{len(dataset)} samples in dataset.")
    run_outs = []
    for n_run in range(5):
        for i, dp in enumerate(loader):
            if i >= args.n_show:
                break
            outs = [i]
            try:
                outs.append(dp["text"][0])
            except KeyError:
                pass
            print(", ".join(str(o) for o in outs))

            # test if dataset is deterministic
            if n_run == 0:
                run_outs.append(outs)
            elif outs != run_outs[i]:
                print(f"Dataset is not deterministic: Run {n_run} {outs} != {run_outs[i]}")
        print()


if __name__ == "__main__":
    main()
