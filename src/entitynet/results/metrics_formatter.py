"""
Formatting kwargs are designed similarly to kwargs for
pandas/io/formats/style_render.py StyleRenderer.format

"""

import math
from copy import deepcopy
from typing import Callable

from attrs import define

from packg import Const
from packg.log import logger


class MetricTypeC(Const):
    FLOAT = "float"  # 3 decimals
    PERCENT = "percent"  # multiply by 100 and then 2 decimals
    INT = "int"  # int
    UNKNOWN = "unknown"  # treat as float or string if it's not a number.


FORMAT_DEFS = {
    MetricTypeC.FLOAT: {
        "formatter": None,
        "precision": 3,
        "thousands": None,
        "multiplier": None,
    },
    MetricTypeC.PERCENT: {
        "formatter": None,
        "precision": 2,
        "thousands": None,
        "multiplier": 100,
    },
    MetricTypeC.INT: {
        "formatter": None,
        "precision": 0,
        "thousands": None,
        "multiplier": None,
    },
}

FORMAT_DEFS_T = dict[str, dict[str, any]] | None


@define
class Formatter:
    formatter: Callable | None = None
    precision: int = 3
    thousands: str | None = None
    multiplier: int | None = None

    def format(self, x) -> str:
        if self.formatter is not None:
            return self.formatter(x)
        x_out = x
        if self.multiplier is not None:
            x_out = x_out * self.multiplier
        if self.precision is not None:
            x_out = round(x_out, self.precision)
        if self.thousands is None:
            fmt_str = "{:." + str(self.precision) + "f}"
            return fmt_str.format(x_out)
        else:
            fmt_str = "{:,." + str(self.precision) + "f}"
            try:
                fmted_str = fmt_str.format(x_out)
            except ValueError as e:
                raise ValueError(f"Format specifier was '{fmt_str}' value was '{x_out}'") from e
            return fmted_str.replace(",", self.thousands)

    def format_to_float(self, x) -> float:
        x_out = x
        if self.multiplier is not None:
            x_out = x_out * self.multiplier
        if self.precision is not None:
            x_out = round(x_out, self.precision)
        return x_out


def get_updated_format_defs(format_defs: FORMAT_DEFS_T) -> FORMAT_DEFS_T:
    final_format_defs = deepcopy(FORMAT_DEFS)
    if format_defs is None:
        return final_format_defs
    for k, v in format_defs.items():
        if k not in FORMAT_DEFS:
            logger.warning(f"Unknown metric type {k} added to formatting definitions.")
            final_format_defs[k] = v
            continue
        final_format_defs[k].update(v)
    return final_format_defs


def get_format_dict_for_metric_type(typ: str, format_defs: FORMAT_DEFS_T = None) -> FORMAT_DEFS_T:
    final_format_defs = get_updated_format_defs(format_defs)
    if typ not in final_format_defs:
        raise ValueError(f"Unknown metric type '{typ}' not in {list(final_format_defs.keys())}")
    format_dict = final_format_defs[typ]
    return format_dict


def get_formatter_for_metric_type(typ: str, format_defs: FORMAT_DEFS_T = None) -> Formatter:
    format_dict = get_format_dict_for_metric_type(typ, format_defs)
    formatter = Formatter(**format_dict)
    return formatter


def format_metric(name: str, value: float, format_defs: FORMAT_DEFS_T = None) -> str:
    final_format_defs = get_updated_format_defs(format_defs)
    typ = get_type_for_metric(name)
    if typ == MetricTypeC.UNKNOWN:
        logger.warning(f"Unknown metric type for {name} assuming float")
        typ = MetricTypeC.FLOAT
    formatter = get_formatter_for_metric_type(typ, final_format_defs)
    try:
        value = float(value)
    except ValueError:
        logger.error(f"Could not convert '{value}' {type(value)} to float")
        return str(math.nan)
    return formatter.format(value)


def get_type_for_metric(name: str):
    if any(name.endswith(a) for a in ["loss", "_meanr"]):
        typ = MetricTypeC.FLOAT
    elif any(
        name.endswith(a)
        for a in [
            "acc1",
            "acc5",
            "acc1_macro",
            "acc5_macro",
            "_r1",
            "_r5",
            "_r10",
            "_r20",
            "_r50",
            "_r100",
            "_r200",
            "_r500",
            "_r1000",
            "_bbaf",
            "_ap",
        ]
    ):
        typ = MetricTypeC.PERCENT
    elif any(name.endswith(a) for a in ["_n", "_medr"]):
        typ = MetricTypeC.INT
    elif name.startswith("ap_"):
        # average precision, but it has already been multiplied by 100
        typ = MetricTypeC.FLOAT
    elif name in [
        "imgn_1k_val",
        "retrieval_avg",
        "domainshift_avg",
        "inat21laten_val",
        "cubobj_test",
        "rarespecies_train",
    ]:
        typ = MetricTypeC.PERCENT
    else:
        typ = MetricTypeC.UNKNOWN
    return typ
