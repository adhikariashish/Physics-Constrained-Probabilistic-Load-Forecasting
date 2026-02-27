from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from src.config.schema import DataCfg


@dataclass(frozen=True)
class SplitCounts:
    train: int
    val: int
    test: int


def _parse_date(s: str) -> pd.Timestamp:
    # accept "YYYY-MM-DD"
    ts = pd.to_datetime(s, errors="raise")
    # normalize to midnight for consistent comparisons
    return ts.normalize()


def _slice_by_date_range(
    df: pd.DataFrame,
    ts_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Inclusive start and inclusive end (date-based). Since our timestamps include hour,
    we treat end as end-of-day by adding 1 day and using < next_day.
    """
    end_exclusive = end + pd.Timedelta(days=1)
    m = (df[ts_col] >= start) & (df[ts_col] < end_exclusive)
    return df.loc[m].copy()


def build_time_splits(
    df: pd.DataFrame,
    data_cfg: DataCfg,
    *,
    ts_col: str = "timestamp",
) -> Tuple[Dict[str, Dict[str, Any]], SplitCounts]:
    """
    Build deterministic time splits from config.

    Expects config structure:
      data.split.strategy: "time"
      data.split.time.train.start/end, val.start/end, test.start/end
    """
    split_cfg = getattr(data_cfg, "split", None)
    if split_cfg is None:
        raise ValueError("data.split is missing in config.")

    strategy = getattr(split_cfg, "strategy", None)
    if (strategy or "").lower() != "time":
        raise ValueError(f"Only time split is supported here. Got split.strategy={strategy!r}")

    time_cfg = getattr(split_cfg, "time", None)
    if time_cfg is None:
        raise ValueError("data.split.time is missing in config.")

    def _get_range(node) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start = _parse_date(getattr(node, "start"))
        end = _parse_date(getattr(node, "end"))
        if end < start:
            raise ValueError(f"Invalid split range: end < start ({start}..{end})")
        return start, end

    tr0, tr1 = _get_range(getattr(time_cfg, "train"))
    va0, va1 = _get_range(getattr(time_cfg, "val"))
    te0, te1 = _get_range(getattr(time_cfg, "test"))

    # Non-overlap sanity checks (date level)
    if not (tr1 < va0 and va1 < te0):
        raise ValueError(
            "Splits must be non-overlapping and ordered: train < val < test.\n"
            f"train: {tr0.date()}..{tr1.date()}, val: {va0.date()}..{va1.date()}, test: {te0.date()}..{te1.date()}"
        )

    train_df = _slice_by_date_range(df, ts_col, tr0, tr1)
    val_df = _slice_by_date_range(df, ts_col, va0, va1)
    test_df = _slice_by_date_range(df, ts_col, te0, te1)

    # Ensure no overlaps in actual timestamps
    train_ts = set(train_df[ts_col].tolist())
    val_ts = set(val_df[ts_col].tolist())
    test_ts = set(test_df[ts_col].tolist())
    if train_ts & val_ts or train_ts & test_ts or val_ts & test_ts:
        raise ValueError("Split timestamp overlap detected. Check ranges and timestamp granularity.")

    splits = {
        "strategy": "time",
        "timezone_note": f"Stored as naive local time; interpret as {getattr(data_cfg.data_schema, 'timezone', 'America/Chicago')}",
        "freq": getattr(data_cfg.data_schema, "freq", "H"),
        "train": {"start": str(tr0.date()), "end": str(tr1.date())},
        "val": {"start": str(va0.date()), "end": str(va1.date())},
        "test": {"start": str(te0.date()), "end": str(te1.date())},
    }

    counts = SplitCounts(train=len(train_df), val=len(val_df), test=len(test_df))
    return splits, counts


def save_splits_json(
    splits: Dict[str, Any],
    counts: SplitCounts,
    out_path: str | Path,
    *,
    extra: Dict[str, Any] | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = dict(splits)
    payload["counts"] = {"train": counts.train, "val": counts.val, "test": counts.test}
    if extra:
        payload.update(extra)

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path