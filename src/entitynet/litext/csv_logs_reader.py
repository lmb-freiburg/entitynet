"""
Read the csv files from the lightning logs and return a dataframe with the metrics.

Note that if there are multiple csvs with conflicting metric values, one of the values just wins
and the others are discarded. To avoid this, keep only the correct csvs and delete the rest.
"""

import logging
from importlib.resources import files
from pathlib import Path

import pandas as pd
from loguru import logger

from packg.log import configure_logger
from packg.typext import PathType
from visiontext.pandatools import full_pandas_display

import entitynet


class CsvLoggerLogsReader:
    COLUMN2PHASE = {
        "train_loss": "train",
        "val_loss": "val",
        "test_loss": "test",
    }

    def __init__(self, run_dir: PathType, report_dups: bool = True):
        self.report_dups = report_dups
        self.output_per_csv = None
        self.output_total = None
        self.read_dir(run_dir)

    def read_dir(self, run_dir: PathType):
        csvs = list((Path(run_dir) / "lightning_logs").glob("version_*/metrics.csv"))
        if len(csvs) == 0:
            logger.error(f"CsvLoggerLogsReader: No csv files found in {run_dir}")
            return
        logger.debug(f"Found {len(csvs)} csv files in {run_dir}")

        output = {phase: {} for phase in self.COLUMN2PHASE.values()}
        for csvfile in csvs:
            logfile_version = csvfile.parent.name
            df = pd.read_csv(csvfile)
            # sort outputs into train, val, test
            for column, phase in self.COLUMN2PHASE.items():
                if column not in df:
                    continue
                idx_col = df[column].notna()
                df_phase = df[idx_col]
                if len(df_phase) == 0:
                    continue
                output[phase][logfile_version] = df_phase

        output_total = {}
        for phase, phase_collected in output.items():
            n_rows = 0
            df_list = []
            for logfile_version, df in phase_collected.items():
                n_rows += len(df)
                df_sorted = df.sort_values(by=["epoch", "step"])
                df_list.append(df_sorted)
            if len(df_list) == 0:
                output_total[phase] = None
                continue
            df_concat = pd.concat(df_list)
            df_nodup = df_concat.drop_duplicates()
            df_final_sorted = df_nodup.sort_values(by=["epoch", "step"])
            df_final_nona = df_final_sorted.dropna(axis=1, how="all")
            logger.debug(f"{phase=}: Got {n_rows} rows total in {len(phase_collected)} dataframes")

            # find mismatches (epoch, step are same, but metrics are different)
            # the groupby.size creates a dataframe that has index epoch, step and column size
            # that we can turn into a multiindex
            grouped = df_final_nona.groupby(["epoch", "step"]).size()
            duplicates = grouped[grouped > 1]
            duplicated_index = duplicates.index

            # change the index of the dataframe and use .loc to apply the index
            df_final_indexed = df_final_nona.set_index(["epoch", "step"])
            df_dups = df_final_indexed.loc[duplicated_index]
            if len(df_dups) > 0:
                if self.report_dups:
                    logger.error(f"Found {len(df_dups)} duplicates in {phase} phase!")
                    # with full_pandas_display():
                    #     print(df_dups)
                    #     print()
                df_final_clean = df_final_indexed.reset_index().drop_duplicates(
                    subset=["epoch", "step"], keep="first"
                )
            else:
                df_final_clean = df_final_nona

            output_total[phase] = df_final_clean
        self.output_per_csv = output
        self.output_total = output_total

    def get_epoch_metric(self, phase: str, epoch: int, metric: str):
        if phase not in self.output_total:
            raise KeyError(f"Phase {phase} not found in {list(self.output_total.keys())}")
        df_phase = self.output_total[phase]
        if df_phase is None:
            return None
        if metric not in df_phase:
            return None
        df_sel = df_phase[(df_phase["epoch"] == epoch) & (df_phase[metric].notna())]
        if len(df_sel) == 0:
            return None
        if len(df_sel) > 1:
            logger.error(f"Found multiple rows for {phase} {epoch} {metric} using first row.")
            with full_pandas_display():
                print(df_sel)
        df_sel = df_sel.iloc[0]
        value = df_sel[metric]
        return float(value)


def main():
    configure_logger(level=logging.DEBUG)
    csvllr = CsvLoggerLogsReader(files(entitynet) / "testdata/example_run")
    for phase, df_total in csvllr.output_total.items():
        if df_total is None:
            print(f"{phase=} has no data")
            continue
        print(f"{phase=}: {df_total.shape}")
        for col in df_total.columns:
            print(f"    {col} (na: {df_total[col].isna().sum()}/{len(df_total)})")
    phase = "val"
    metric = "val_loss"
    value = csvllr.get_epoch_metric(phase, 0, metric)
    print(f"{phase=} {metric=} {value=:.3f}")


if __name__ == "__main__":
    main()
