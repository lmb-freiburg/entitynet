"""
folder structure
    experiments_dir/project_name/experiment_name/run_name/
"""

import math
import re
from json import JSONDecodeError
from math import isnan
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from natsort import natsorted

from packg import Const, format_exception
from packg.iotools import load_json, make_git_pathspec, yield_lines_from_file
from packg.typext import PathType

from entitynet.results.checkpoint_finder import find_checkpoints

ExperimentTuple = tuple[Path, str, str, str, str, pd.DataFrame]

DEFAULT_METRICS_FILTER_STR = "*acc1,*_r1,*_r5,*_meanr,*all_ap,*ap_all,*loss"
DEFAULT_METRICS_FILTER_LIST = DEFAULT_METRICS_FILTER_STR.split(",")
# TODO unify id_columns, in case they are also defined elsewhere.
ID_COLUMNS = ["project", "experiment", "run", "neptune_id", "phase", "epoch", "step"]
ID_COLUMNS_SET = set(ID_COLUMNS)


def find_runs_by_runconfig_yamls(
    experiments_dir: PathType,
    subfolder: str,
    skip_empty: bool = True,
    experiment_filter_list: list[str] | None = None,
) -> list[Path]:
    """
    Find all runs for this subfolder (i.e. project) by looking for runconfig.yaml files.
    """
    experiments_dir = Path(experiments_dir)
    spec = None if experiment_filter_list is None else make_git_pathspec(experiment_filter_list)
    expected_depth = 3
    exptected_values = "project", "experiment", "run"
    glob_str = f"{subfolder}/**/runconfig.yaml"
    config_yamls = natsorted(list(experiments_dir.glob(glob_str)))
    run_dirs_rel = []
    for config_yaml in config_yamls:
        cand_dir = config_yaml.parent
        cand_dir_rel = cand_dir.relative_to(experiments_dir)
        depth = len(cand_dir_rel.parts)
        if depth != expected_depth:
            logger.error(
                f"Could not read experiment! Expected {expected_depth} subfolders with values for "
                f"{', '.join(exptected_values)} but got {depth} subfolders: {cand_dir_rel}"
            )
            continue
        if spec is not None and not spec.match_file(cand_dir_rel):
            continue
        if skip_empty and check_run_is_empty(cand_dir):
            continue
        run_dirs_rel.append(cand_dir_rel)
    return run_dirs_rel


def check_run_is_empty(run_dir: PathType):
    # 1. check if any output json files exist
    found_train_output, found_test_only_output = get_experiment_status_from_results(run_dir)
    if found_train_output or found_test_only_output:
        return False

    # 2. check if last.json exists
    if (Path(run_dir) / "ckpt" / "last.json").is_file():
        return False

    # 3. get experiment status from checkpoints
    found_train_ckpts, found_test_only_output, last_ckpt, best_ckpt = (
        get_experiment_status_from_checkpoints(run_dir, log_errors=False)
    )
    if found_train_ckpts or found_test_only_output:
        return False

    return True


def load_neptune_id_for_run(run_dir: PathType) -> str | None:
    """Collect neptune id if exists."""
    neptune_id_file = run_dir / "neptune_id.txt"
    neptune_id = None
    if neptune_id_file.is_file():
        neptune_ids = list(yield_lines_from_file(neptune_id_file))
        neptune_ids = [nid for nid in neptune_ids if str(nid).lower() != "none"]
        if len(set(neptune_ids)) > 1:
            logger.error(
                f"Multiple neptune ids found in {neptune_id_file}: {neptune_ids} - "
                f"returning the last one."
            )
        if len(neptune_ids) == 0:
            raise ValueError(f"Corrupt neptune id file: {neptune_id_file}")
        neptune_id = neptune_ids[-1]
    return neptune_id


RE_OUTPUT_FILE = re.compile(r"^(.*?)_ckpt-(\d+)-(\d+)_task-([a-zA-Z0-9_-]+)-results\.json$")


def load_eval_results_for_run(
    experiments_dir: PathType,
    run_dir_rel: PathType,
    filter_dict: dict[str, list[str] | None] | None = None,
    allow_nans: bool = True,
    allow_nans_always: tuple[str] | None = (),
    datasets: list[str] | None = None,
    remove_corrupt_jsons: bool = False,
) -> pd.DataFrame | None:
    """
    For this run, find all *results.json files and load them into dataframes.

    Args:
        experiments_dir: base output directory
        run_dir_rel: subfolder in format project_name/experiment_name/run_name/
        filter_dict: dictionary column_name -> list of values to filter in gitignore format.
            input columns are "phase", "epoch", "step", "dataset", "metric".
            so to view only the test phase, set filter_dict={"phase": ["test"]}.
            empty or none filter list will be ignored.
        allow_nans:
        allow_nans_always:
        remove_corrupt_jsons: whether to remove corrupt json

    Returns:

    """
    experiments_dir = Path(experiments_dir)
    run_dir_rel = Path(run_dir_rel)
    run_dir = experiments_dir / run_dir_rel
    glob_strs = ["*results.json"]
    if datasets is not None:
        glob_strs = [f"*_ckpt-*-*_task-{ds}-results.json" for ds in datasets]
    files = []
    for glob_str in glob_strs:
        files += list((run_dir / "outputs").glob(glob_str))
    files = natsorted(files)

    filter_specs = {}
    if filter_dict is not None:
        for filter_column, filter_list in filter_dict.items():
            if isinstance(filter_list, str):
                raise ValueError(
                    f"values of filter_dict must be list, got string "
                    f"'{filter_list}' for key '{filter_column}'"
                )
            if filter_list is None or len(filter_list) == 0:
                continue
            filter_list = list(map(str, filter_list))  # accept e.g. epoch as int and convert to str
            filter_spec = make_git_pathspec(filter_list)
            filter_specs[filter_column] = filter_spec

    # build the spec where we have to allow some nans always, for some metrics
    spec_nan_ok = None
    if allow_nans_always is not None:
        spec_nan_ok = make_git_pathspec(allow_nans_always)

    # go through all results jsons and collect their data into a dataframe
    data_lines = []
    for file in files:
        re_match = RE_OUTPUT_FILE.match(file.name)
        if re_match is None:
            logger.error(f"Cannot understand filename: {file}")
            continue
        phase, epoch_str, step_str, dataset_name = re_match.groups()
        epoch = int(epoch_str)
        step = int(step_str)

        # here we already apply the filter to skip loading jsons that we don't need.
        kv_to_filter = {
            "phase": phase,
            "epoch": epoch_str,
            "step": step_str,
            "dataset": dataset_name,
        }
        skip = False
        for col_name, col_value in kv_to_filter.items():
            if col_name in filter_specs:
                if not filter_specs[col_name].match_file(col_value):
                    skip = True
                    break
                else:
                    pass
        if skip:
            continue

        # load and parse the evaluation result json
        try:
            results = load_json(file)
        except JSONDecodeError as e:
            if remove_corrupt_jsons:
                logger.error(f"Removing {file} due to {format_exception(e)}")
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove {file} due to {format_exception(e)}")
            else:
                logger.error(f"Could not load {file} due to {format_exception(e)}")

            continue
        for metric_key, metric_value in results.items():
            # problem is that some of the metrics can be nan and it's not wrong.
            # e.g. average precision metric, when there is no data for them,
            # and later we want to do np.nanmean to ignore them. so setting them to 0 is also bad.
            if not allow_nans and (metric_value is None or isnan(metric_value)):
                # nan detected, and not allowed
                if spec_nan_ok is not None and spec_nan_ok.match_file(metric_key):
                    # nan detected, and not allowed, but still OK for this metric.
                    pass
                else:
                    raise ValueError(
                        f"NaN found, not allowed, and spec nan ok did not match:\n"
                        f"{spec_nan_ok=}\n"
                        f"Metric {metric_key} {metric_value}\n"
                        f"{file}\n"
                    )

            # the dataset name is included in the metric, but we don't want it anymore at this point
            # but rather have dataset as separate column.
            metric_key_new = metric_key.split(dataset_name)[-1].lstrip("_")

            if "metric" in filter_specs and not filter_specs["metric"].match_file(metric_key_new):
                continue

            data_lines.append((phase, epoch, step, dataset_name, metric_key_new, metric_value))

    if len(data_lines) == 0:
        logger.info(f"No data found for {run_dir}")
        df = None
    else:
        df = pd.DataFrame(
            data_lines, columns=["phase", "epoch", "step", "dataset", "metric", "value"]
        )
    return df


def merge_metrics_to_columns(
    expe_base_dir, list_run_dir_rel: list[PathType], list_run_df
) -> pd.DataFrame:
    """
    2. condense layout of the dataframe such that metrics are columns
    3. if neps, also add the hyperparameters as columns
    """
    expe_base_dir = Path(expe_base_dir)

    # merge all dataframes into one
    header_run = ["project", "experiment", "run", "neptune_id", "phase", "epoch", "step"]
    header_metric = ["dataset", "metric", "value"]
    header = header_run + header_metric
    all_data = []
    for run_dir_rel, run_df in zip(list_run_dir_rel, list_run_df):
        project, experiment, run = Path(run_dir_rel).parts
        run_dir = expe_base_dir / Path(run_dir_rel)
        neptune_id = load_neptune_id_for_run(run_dir)

        if run_df is None:
            logger.warning(f"No data found for {project}/{experiment}/{run}")
            continue
        assert run_df.shape[0] > 0, (
            f"Empty dataframe {run_df} with shape {run_df.shape} for "
            f"{project} / {experiment} / {run} - this should be None instead "
        )

        for row in run_df.itertuples(index=False):
            all_data.append([project, experiment, run, neptune_id] + list(row))
    df_all = pd.DataFrame(all_data, columns=header)

    # instead of one row per dataset and metric, create one column per dataset and metric
    dataset_metric_tuples = list(zip(df_all["dataset"], df_all["metric"]))
    sep = "/"
    dataset_metric_strs = [f"{ds}{sep}{mt}" for ds, mt in dataset_metric_tuples]
    metric_column_names = sorted(set(dataset_metric_strs))
    new_data_dict = {}
    for (
        project,
        experiment,
        run,
        neptune_id,
        phase,
        epoch,
        step,
        dataset,
        metric,
        val,
    ) in all_data:
        result_key = (project, experiment, run, neptune_id, phase, epoch, step)
        if result_key not in new_data_dict:
            new_data_dict[result_key] = {n: math.nan for n in metric_column_names}
        metric_column = f"{dataset}{sep}{metric}"
        new_data_dict[result_key][metric_column] = val
    new_header = header_run + metric_column_names
    new_data_lines = []
    for result_key, metric_dict in new_data_dict.items():
        new_data_lines.append(list(result_key) + list(metric_dict.values()))
    df_metrics_as_cols = pd.DataFrame(new_data_lines, columns=new_header)
    return df_metrics_as_cols


class ActionOnMissingColumn(Const):
    ADD = "add"
    IGNORE = "ignore"
    RAISE = "raise"


def sort_and_filter_columns_in_result_df(
    result_df: pd.DataFrame,
    columns: list[str],
    action_on_missing_column: str = ActionOnMissingColumn.ADD,
) -> pd.DataFrame:
    """
    Given a dataframe with one row as one run, sort and filter the columns.

    Args:
        result_df: input dataframe with one row as one run.
        columns: target columns that should be in the output dataframe
            probably you want some of the identifier columns:
            "project", "experiment", "run", "neptune_id", "phase", "epoch", "step"
            and then some metric columns like "dataset_split/metric"
        action_on_missing_column: what to do if the column is not in the input dataframe.
            "add" an empty column, "ignore" the column, "raise" an error.

    Returns:
        sorted and filtered dataframe
    """
    assert len(columns) == len(set(columns)), f"Columns must be unique but got {columns}"
    # added all the copying to remove the SettingWithCopyWarning:
    # A value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_indexer,col_indexer] = value instead
    temp_df = result_df.copy()
    actual_columns = []
    for column in columns:
        if column not in temp_df.columns:
            match action_on_missing_column:
                case ActionOnMissingColumn.ADD:
                    temp_df[column] = np.nan
                case ActionOnMissingColumn.IGNORE:
                    continue
                case ActionOnMissingColumn.RAISE:
                    raise ValueError(
                        f"Column {column} not found in the input dataframe.\n{result_df}"
                    )
        actual_columns.append(column)
    df_sorted = temp_df[actual_columns].copy()
    del temp_df
    return df_sorted


def get_experiment_status_from_checkpoints(
    run_dir: PathType, log_errors: bool = True
) -> tuple[bool, bool, dict | None, dict | None]:
    """
    For a given experiment, figure out it's status based on the saved checkpoints.

    Args:
        run_dir: /path/crossmodal_output/experiments
        log_errors: whether to log errors
    """
    run_dir = Path(run_dir)

    # check if checkpoints exist and if yes, how far the experiment got
    found_train_ckpts, found_test_only_output, last_ckpt, best_ckpt = False, False, None, None
    ckpt_dir = run_dir / "ckpt"
    if not ckpt_dir.is_dir():
        logger.debug(f"No ckpt dir found: {ckpt_dir}")
    else:
        last_ckpt, best_ckpt, _ = find_checkpoints(ckpt_dir, log_errors=log_errors)
        if last_ckpt == best_ckpt:
            last_ckpt = None
        if last_ckpt is None and best_ckpt is None:
            logger.debug(f"No checkpoints found in {ckpt_dir}")
        else:
            found_train_ckpts = True
            logger.debug(f"Found checkpoints {last_ckpt} and {best_ckpt}")

    # check if test-only output exists
    output_dir = run_dir / "outputs"
    test_files = list(output_dir.glob("test_ckpt-0-0_*.json"))
    if len(test_files) > 0:
        found_test_only_output = True
    return found_train_ckpts, found_test_only_output, last_ckpt, best_ckpt


def get_experiment_status_from_results(run_dir: PathType) -> tuple[bool, bool]:
    """
    For a given experiment, figure out it's status based on the JSON result outputs.
    """
    run_dir = Path(run_dir)

    # check if val outputs exist
    output_dir = run_dir / "outputs"
    if not output_dir.is_dir():
        return False, False
    val_output_files = list(output_dir.glob("val_ckpt-*.json"))
    val_epochs = {}
    for val_output_file in val_output_files:
        val_epoch = int(val_output_file.name.split("-")[1])
        val_epochs[val_epoch] = True
    found_train_output = len(val_epochs) > 0

    # check if test-only output exists
    test_files = list(output_dir.glob("test_ckpt-0-0_*.json"))
    found_test_only_output = len(test_files) > 0
    return found_train_output, found_test_only_output
