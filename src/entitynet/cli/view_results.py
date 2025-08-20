"""
View metrics for experiments.

No aggregation yet.
"""

import getpass
import os
from pathlib import Path
from pprint import pformat
from typing import Optional

import pandas as pd
from attrs import define
from loguru import logger

from packg.dtime import get_timestamp_for_filename
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from visiontext.pandatools import full_pandas_display

from entitynet.miscutils import dump_dataframe
from entitynet.paths import get_entitynet_output_dir
from entitynet.results.metrics_formatter import get_formatter_for_metric_type, get_type_for_metric
from entitynet.results.results_loader import (
    DEFAULT_METRICS_FILTER_STR,
    ExperimentTuple,
    find_runs_by_runconfig_yamls,
    load_eval_results_for_run,
    merge_metrics_to_columns,
    sort_and_filter_columns_in_result_df,
)
from entitynet.results.results_viewer_utils import (
    DEFAULT_OUTPUT_COLUMNS_KEY,
    OUTPUT_COLUMNS_DICT,
    RENAME_COLUMNS,
)


@define
class Args(VerboseQuietArgs):
    subfolder: Optional[str] = add_argument(
        shortcut="-s", type=str, help="Subfolder of experiments to check", default=None
    )
    search_experiment: str = add_argument(
        shortcut="-e", type=str, help="Show experiments with this string in the name", default="*"
    )
    filter_phase: str | None = add_argument(shortcut="-p", help="Phase to show", default=None)
    filter_metric: str = add_argument(shortcut="-m", default=DEFAULT_METRICS_FILTER_STR)
    filter_dataset: str = add_argument(shortcut="-d", default="*")
    display_raw_data: bool = add_argument(
        shortcut="-r", action="store_true", help="Display all raw data in console"
    )
    columns: str = add_argument(
        default=DEFAULT_OUTPUT_COLUMNS_KEY,
        help="Columns to show",
        choices=list(OUTPUT_COLUMNS_DICT.keys()),
    )
    mode: str = add_argument(
        shortcut="-o",
        default="bestlast",
        choices=["all", "zero", "bestlast", "best", "last"],
        help="Epoch filtering mode",
    )


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    # determine filename to dump the files
    user = getpass.getuser()
    if Path(f"/ihome/{user}").is_dir():
        out_path = f"/ihome/{user}/temp_results_output"
    else:
        out_path = f"/home/{user}/temp_results_output"
    filename = f"{args.subfolder}/results_{get_timestamp_for_filename()}"
    outputs_to_print = []

    # find base directory to search for experiments
    expe_base_dir = get_entitynet_output_dir() / "experiments"
    subfolders = sorted(x for x in os.listdir(expe_base_dir) if (expe_base_dir / x).is_dir())
    if args.subfolder is None:
        logger.error(f"Set -s to one of the subfolders here: {subfolders}")
        return
    subfolders = args.subfolder.split(",")
    list_run_dir_rel = []
    for subfolder in subfolders:
        target_dir = expe_base_dir / subfolder
        assert target_dir.is_dir(), f"Directory {target_dir} does not exist"
        list_run_dir_rel_here = find_runs_by_runconfig_yamls(
            expe_base_dir,
            subfolder,
            skip_empty=True,
            experiment_filter_list=args.search_experiment.split(","),
        )
        list_run_dir_rel.extend(list_run_dir_rel_here)
    logger.info(f"Found {len(list_run_dir_rel)} candidates when searching for runconfig.yaml")

    # build filters for each column and merge the dataframe such that metrics are columns
    filter_args = {
        "metric": args.filter_metric,
        "dataset": args.filter_dataset,
        "phase": args.filter_phase,
    }
    filter_dict = {k: v.split(",") for k, v in filter_args.items() if v is not None and v != ""}
    logger.info(f"Filters: {pformat(filter_dict)}")

    list_run_df = [
        load_eval_results_for_run(expe_base_dir, run_dir_rel, filter_dict=filter_dict)
        for run_dir_rel in list_run_dir_rel
    ]
    _ = ExperimentTuple  # this script wants a experiment tuple here

    if args.display_raw_data:
        for run_dir_rel, run_df in zip(list_run_dir_rel, list_run_df):
            print(f"{run_dir_rel}")
            print(run_df)
            print()

    df_metrics_as_cols = merge_metrics_to_columns(expe_base_dir, list_run_dir_rel, list_run_df)
    outputs_to_print += dump_dataframe(df_metrics_as_cols, out_path, filename + "_unsorted")
    if args.display_raw_data:
        with full_pandas_display():
            print(df_metrics_as_cols)
            print()

    # # also dump the completely unfiltered data
    # df_unfiltered, df_unfiltered_neps = rrl.merge_and_filter_experiments(experiments, {})
    # outputs_to_print += dump_dataframe(df_unfiltered, out_path, filename + "_full")
    # print("\n".join(df_unfiltered.columns.tolist()))

    # filter and sort columns again
    columns = OUTPUT_COLUMNS_DICT[args.columns]
    df_sorted = sort_and_filter_columns_in_result_df(df_metrics_as_cols, columns)
    for column in df_sorted.columns:
        metric_type = get_type_for_metric(column)
        print(f"{column}: {metric_type}")
        if metric_type == "unknown":
            continue
        formatter = get_formatter_for_metric_type(metric_type)
        df_sorted[column] = df_sorted[column].apply(formatter.format).apply(float)

    for name_before, name_after in RENAME_COLUMNS.items():
        if name_before in df_sorted.columns:
            df_sorted.rename(columns={name_before: name_after}, inplace=True)

    if args.mode != "all":
        group_cols = ["project", "experiment", "run"]
        epoch_counts = df_sorted.groupby(group_cols)["epoch"].nunique()
        multi_epoch_combinations = epoch_counts[epoch_counts > 1].index
        keep_mask = pd.Series(True, index=df_sorted.index)

        for combo in multi_epoch_combinations:
            if isinstance(combo, tuple):
                combo_mask = pd.Series(True, index=df_sorted.index)
                for i, col in enumerate(group_cols):
                    combo_mask &= df_sorted[col] == combo[i]
            else:
                combo_mask = df_sorted[group_cols[0]] == combo

            # assuming the test is only run on the zero, best, and last epoch, we can assume
            # the epochs we have here are (0, best, last) or (0, best=last)
            combo_epochs = sorted(df_sorted[combo_mask]["epoch"].unique().tolist())
            best_epoch = combo_epochs[-2] if len(combo_epochs) > 1 else combo_epochs[0]
            last_epoch = combo_epochs[-1]

            if args.mode == "zero":  # show only epoch zero
                epochs_to_remove = [e for e in combo_epochs if e != 0]
            elif args.mode == "bestlast":
                epochs_to_remove = [e for e in combo_epochs if e != best_epoch and e != last_epoch]
            elif args.mode == "best":
                epochs_to_remove = [e for e in combo_epochs if e != best_epoch]
            elif args.mode == "last":
                epochs_to_remove = [e for e in combo_epochs if e != last_epoch]

            for epoch_to_remove in epochs_to_remove:
                epoch_mask = combo_mask & (df_sorted["epoch"] == epoch_to_remove)
                keep_mask &= ~epoch_mask

        df_sorted = df_sorted[keep_mask].copy()
        logger.info(f"Applied {args.mode} filter: kept {len(df_sorted)} rows")

    outputs_to_print += dump_dataframe(df_sorted, out_path, filename + "_sorted")

    with full_pandas_display():
        print(df_sorted)
        print()

    for o in outputs_to_print:
        print(o)


if __name__ == "__main__":
    main()
