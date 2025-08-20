"""
Run train dataloader from config, to verify speed and correctness.

Note: This ignores multi-gpu setups
"""

from pathlib import Path
from timeit import default_timer

from attrs import asdict, define
from loguru import logger

from packg.debugging import connect_to_pycharm_debug_server
from packg.iotools import dumps_yaml
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.tqdmext import tqdm_max_ncols
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from typedparser.objects import repr_value
from visiontext.distutils import get_world_info
from visiontext.profiling import start_pyinstrument_profiler, stop_pyinstrument_profiler

from entitynet.config.config_factory import load_config_from_file
from entitynet.config.main_config import Config
from entitynet.paths import get_entitynet_data_dir, get_entitynet_output_dir
from entitynet.tasks.task_factory import build_train_and_val_tasks


@define
class Args(VerboseQuietArgs):
    config_file: Path = add_argument("config_file", type=str, help="Experiment config file")
    options: list[str] | None = add_argument(shortcut="-o", action="append", help="Override config")
    trace: str | None = add_argument(type=str, help="Connect debug server on this host.")
    trace_port: int = add_argument(type=int, default=33553, help="Target debugging server port")
    workers: int | None = add_argument(type=int, default=None, help="Overwrite worker settings")
    partial_dataloader: bool = add_argument(
        action="store_true", help="Run partial train dataloader to profile"
    )
    full_dataset: bool = add_argument(action="store_true", help="Run full train dataset to profile")
    full_dataloader: bool = add_argument(
        action="store_true", help="Run full train dataloader to profile"
    )


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")
    global_rank, world_size = get_world_info()
    if args.trace is not None and global_rank == 0:
        connect_to_pycharm_debug_server(args.trace, args.trace_port)
    logger.info(f"dataset dir: {get_entitynet_data_dir()}")
    logger.info(f"crx output dir: {get_entitynet_output_dir()}")

    config = load_config_from_file(args.config_file, merge_dotlist=args.options)
    config.trainer.val_task_keys = []
    config.trainer.test_task_keys = []
    config.eval_tasks = {}

    if args.workers is not None:
        config.trainer.workers = args.workers
    train_task, train_dataset, train_dataloader, _, _, _, _, _ = build_train_and_val_tasks(
        config, {}, {}, world_size=1
    )

    print(f"{len(train_dataset)} samples in dataset.")

    start_pyinstrument_profiler()
    if args.full_dataset:
        run_full_dataset(train_dataset)
    elif args.full_dataloader:
        run_full_dataloader(train_dataloader)
    elif args.partial_dataloader:
        run_partial_dataloader(train_dataloader)
    else:
        run_partial_dataset(train_dataset)
    stop_pyinstrument_profiler(open_in_browser=False)


def get_dataloader_num_workers(dataloader) -> int:
    try:
        return int(dataloader.num_workers)
    except AttributeError:
        pass
    try:
        return int(dataloader.workers)
    except AttributeError:
        pass
    logger.error(f"Cannot determine number of workers from {dataloader}")
    return 0


def run_full_dataset(train_dataset):
    # go through full train set once
    pbar = tqdm_max_ncols(total=len(train_dataset), desc="full dataset")
    t1 = default_timer()
    n_total = 0
    for batch in train_dataset:
        batch_size = len(batch["image"])
        n_total += batch_size
        pbar.update(batch_size)
    delta = default_timer() - t1
    delta_per = delta / n_total
    print(f"Got {n_total} in {delta:.3f}s per datapoint {delta_per*1000:.3f}ms")


def run_full_dataloader(train_dataloader):
    # go through full dataloader once
    pbar = tqdm_max_ncols(total=len(train_dataloader), desc="full dataloader")
    t1 = default_timer()
    n_total, n_batches = 0, 0
    for batch in train_dataloader:
        batch_size = len(batch["image"])
        n_total += batch_size
        n_batches += 1
        pbar.update()
    delta = default_timer() - t1
    delta_per_datapoint = delta / n_total
    delta_per_batch = delta / n_batches
    workers = get_dataloader_num_workers(train_dataloader)
    print(
        f"Got {n_total} datapoints in {delta:.3f}s per datapoint {delta_per_datapoint*1000:.3f}ms"
    )
    print(f"Got {n_batches} batches in {delta:.3f}s per batch {delta_per_batch*1000:.3f}ms")
    print(f"With {workers} workers.")


_N_WARMUP = 10  # warmup iterations to let disks spin up, etc.


def run_partial_dataset(train_dataset):
    n_total_datapoints = 1000

    t1 = None
    for i, datapoint in enumerate(
        tqdm_max_ncols(train_dataset, desc="dataset", total=n_total_datapoints + _N_WARMUP)
    ):
        if i == 0:
            print(repr_value(datapoint))
        if i == _N_WARMUP:
            t1 = default_timer()
        if i >= n_total_datapoints + _N_WARMUP:
            break
    n_total_datapoints = min(n_total_datapoints, len(train_dataset) - _N_WARMUP)
    tdelta = default_timer() - t1
    tdelta_per = tdelta / n_total_datapoints
    logger.info(f"train_dataset (foreground): {tdelta_per*1000:.3f}ms per datapoint.")


def run_partial_dataloader(train_dataloader):
    t1 = None

    print(f"{getattr(train_dataloader, 'num_workers', 'unknown')} number of workers.")
    n_total_batches = 50
    pbar = tqdm_max_ncols(total=n_total_batches + _N_WARMUP, desc="dataloader batches")
    batch_size = None
    for i, batch in enumerate(train_dataloader):
        if batch_size is None:
            batch_size = len(batch["image"])
        pbar.update()
        if i == 0:
            print(repr_value(batch))
        if i == _N_WARMUP:
            t1 = default_timer()
        if i >= n_total_batches + _N_WARMUP:
            break
    pbar.close()
    tdelta = default_timer() - t1
    tdelta_per_batch = tdelta / n_total_batches

    tdelta_per_datapoint = tdelta_per_batch / batch_size
    workers = get_dataloader_num_workers(train_dataloader)
    logger.info(
        f"train_dataloader ({workers} workers): {tdelta_per_batch*1000:.3f}ms per batch at "
        f"{batch_size} batch size / {tdelta_per_datapoint*1000:.3f}ms per datapoint."
    )


if __name__ == "__main__":
    main()
