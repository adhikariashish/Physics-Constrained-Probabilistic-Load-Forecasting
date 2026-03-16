from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

from src.config.schema import DataCfg


# Native Load zones present in your canonical ingestion output
ZONES: List[str] = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST"]
DEFAULT_SYSTEM_COL = "ERCOT"


@dataclass(frozen=True)
class CleanStats:
    # row counts
    rows_in: int
    rows_after_drop_bad_timestamps: int
    rows_after_dedupe: int
    rows_after_missing_policy: int

    # actions taken
    duplicates_dropped: int
    missing_target_dropped: int

    # hourly grid diagnostics
    missing_hours_estimated: int

    # range + metadata
    start_timestamp: str
    end_timestamp: str
    freq: str
    timezone_note: str
    missing_policy: str
    hour_convention_note: str

    # outputs
    rows_out_system: int
    rows_out_zones: int
    save_zones: bool

    # System Hourly Enforcementy
    hourly_expected_hours_system: int
    hourly_missing_inserted_system: int
    hourly_imputed_cells_system: int
    hourly_rows_dropped_after_impute_system: int

    # Zones hourly enforcement
    hourly_expected_hours_zones: int
    hourly_missing_inserted_zones: int
    hourly_imputed_cells_zones: int
    hourly_rows_dropped_after_impute_zones: int

@dataclass(frozen=True)
class HourlyEnforceStats:
    start_ts: str
    end_ts: str
    expected_hours: int
    original_rows: int
    duplicates_dropped: int
    missing_hours_inserted: int
    imputed_cells: int
    rows_dropped_after_impute: int

@dataclass(frozen=True)
class TimeFeatureStats:
    rows_in: int
    rows_out: int
    inferred_freq: Optional[str]
    n_holidays: int
    features_added: List[str]

def _ensure_datetime_naive(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Ensure ts_col is datetime64[ns] and timezone-naive.

    We intentionally keep timestamps naive in Milestone 1 to avoid DST collisions
    and preserve the original ERCOT "Hour Ending" local clock semantics.
    """
    out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(out[ts_col]):
        out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")

    # If tz-aware sneaks in, drop tz info to keep naive local clock time.
    # (We treat it as America/Chicago by definition; stored as metadata.)
    try:
        if getattr(out[ts_col].dt, "tz", None) is not None:
            out[ts_col] = out[ts_col].dt.tz_convert("America/Chicago").dt.tz_localize(None)
    except Exception:
        # If any weirdness, fall back to best-effort naive conversion
        out[ts_col] = pd.to_datetime(out[ts_col].astype(str), errors="coerce")

    return out


def _sort_and_dedupe_keep_last(df: pd.DataFrame, ts_col: str) -> Tuple[pd.DataFrame, int]:
    """
    Deterministic dedupe on timestamp:
      - sort ascending
      - drop duplicates keeping last occurrence
    """
    before = len(df)
    out = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)
    dropped = before - len(out)
    return out, dropped


def _estimate_missing_hours(df: pd.DataFrame, ts_col: str, freq: str) -> int:
    """
    Estimate missing timestamps on an hourly grid between min and max (inclusive).

    NOTE: because we keep timestamps naive local time, DST transitions can produce
    apparent gaps/duplicates depending on the reporting convention. We treat this
    as a diagnostic only (logged).
    """
    if df.empty:
        return 0
    tmin = df[ts_col].min()
    tmax = df[ts_col].max()
    expected = len(pd.date_range(start=tmin, end=tmax, freq=freq))
    actual = len(df)
    return max(0, expected - actual)


def _apply_missing_policy_target(
    df: pd.DataFrame,
    *,
    ts_col: str,
    target_col: str,
    policy: str,
) -> Tuple[pd.DataFrame, int]:
    """
    Apply missingness policy to the target column onl.
    """
    before = len(df)
    pol = (policy or "drop").strip().lower()

    if pol == "drop":
        out = df.dropna(subset=[target_col]).reset_index(drop=True)
        return out, before - len(out)

    if pol == "impute":
        # Minimal deterministic impute: ffill then bfill on target.
        out = df.copy()
        out = out.sort_values(ts_col).reset_index(drop=True)
        out[target_col] = out[target_col].ffill().bfill()
        out = out.dropna(subset=[target_col]).reset_index(drop=True)
        return out, before - len(out)

    raise ValueError(f"Unknown missingness.policy={policy!r}. Expected 'drop' or 'impute'.")

def enforce_hourly_index_hybrid(
    df: pd.DataFrame,
    *,
    data_cfg: DataCfg,
    ts_col: str = "timestamp",
    value_cols: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, HourlyEnforceStats]:
    """
    Enforce a strict hourly index and apply hybrid missingness policy:
      - forward-fill small gaps up to `missingness.impute.limit`
      - drop remaining missing rows (large gaps)
    """
    if value_cols is None:
        value_cols = [c for c in df.columns if c != ts_col]

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    # 1) sort + drop duplicates
    df = df.sort_values(ts_col).reset_index(drop=True)
    dup_mask = df.duplicated(subset=[ts_col], keep="first")
    duplicates_dropped = int(dup_mask.sum())
    df = df.loc[~dup_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("After timestamp cleaning/dedup, dataframe is empty.")

    start_ts = df[ts_col].iloc[0]
    end_ts = df[ts_col].iloc[-1]

    # 2) full hourly index
    full_idx = pd.date_range(start=start_ts, end=end_ts, freq="h")
    expected_hours = int(len(full_idx))

    df = df.set_index(ts_col)

    original_rows = int(len(df))
    df = df.reindex(full_idx)

    missing_hours_inserted = int(df[value_cols].isna().any(axis=1).sum())

    # 3) hybrid impute: ffill small gaps up to limit
    miss_cfg = getattr(data_cfg, "missingness", None)
    impute_cfg = getattr(miss_cfg, "impute", None) if miss_cfg else None

    method = (getattr(impute_cfg, "method", "ffill") if impute_cfg else "ffill").lower()
    limit = int(getattr(impute_cfg, "limit", 3) if impute_cfg else 3)

    if method not in {"ffill"}:
        raise ValueError(f"Only ffill supported in hybrid mode for now. Got impute.method={method!r}")

    before_na = int(df[value_cols].isna().sum().sum())
    df[value_cols] = df[value_cols].ffill(limit=limit)
    after_na = int(df[value_cols].isna().sum().sum())
    imputed_cells = int(before_na - after_na)

    # 4) drop remaining missing rows in value cols (large gaps)
    rows_before_drop = int(len(df))
    df = df.dropna(subset=value_cols)
    rows_dropped_after_impute = int(rows_before_drop - len(df))

    # restore timestamp column
    df = df.reset_index().rename(columns={"index": ts_col})

    stats = HourlyEnforceStats(
        start_ts=str(start_ts),
        end_ts=str(end_ts),
        expected_hours=expected_hours,
        original_rows=original_rows,
        duplicates_dropped=duplicates_dropped,
        missing_hours_inserted=missing_hours_inserted,
        imputed_cells=imputed_cells,
        rows_dropped_after_impute=rows_dropped_after_impute,
    )
    return df, stats

def add_time_features(
    df: pd.DataFrame,
    *,
    data_cfg: DataCfg,
    ts_col: str = "timestamp",
) -> tuple[pd.DataFrame, TimeFeatureStats]:
    """
      data.features.calendar.enabled
      data.features.calendar.hour_of_day.enabled + encoding
      data.features.calendar.day_of_week.enabled + encoding
      data.features.calendar.is_weekend.enabled
      data.features.calendar.is_holiday.enabled + calendar_name

    Outputs columns (when enabled):
      hour_sin, hour_cos
      dow_sin, dow_cos
      is_weekend
      is_holiday
    """

    if ts_col not in df.columns:
        raise KeyError(f"Expected {ts_col!r} in df. Got columns: {df.columns.tolist()}")

    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    rows_in = int(len(df))
    rows_out = int(len(out))

    inferred_freq = None
    try:
        inferred_freq = pd.infer_freq(out[ts_col].iloc[: min(5000, len(out))])
    except Exception:
        inferred_freq = None

    # read config
    feat_cfg = getattr(data_cfg, "features", None)
    cal_cfg = getattr(feat_cfg, "calendar", None) if feat_cfg else None

    if not cal_cfg or not bool(getattr(cal_cfg, "enabled", False)):
        return out, TimeFeatureStats(rows_in, rows_out, inferred_freq, 0, [])

    added: List[str] = []
    ts = out[ts_col]

    hour = ts.dt.hour.astype(int)
    dow = ts.dt.dayofweek.astype(int)  # Mon=0..Sun=6
    is_weekend = (dow >= 5).astype(int)

    def _sin(x: pd.Series, period: int) -> np.ndarray:
        return np.sin(2.0 * np.pi * x.to_numpy(dtype=np.float64) / float(period)).astype(np.float32)

    def _cos(x: pd.Series, period: int) -> np.ndarray:
        return np.cos(2.0 * np.pi * x.to_numpy(dtype=np.float64) / float(period)).astype(np.float32)

    # hour_of_day
    hod_cfg = getattr(cal_cfg, "hour_of_day", None)
    if hod_cfg and bool(getattr(hod_cfg, "enabled", False)):
        enc = str(getattr(hod_cfg, "encoding", "sin_cos")).lower()
        if enc != "sin_cos":
            raise ValueError(f"Unsupported hour_of_day encoding: {enc!r} (expected 'sin_cos')")
        out["hour_sin"] = _sin(hour, 24)
        out["hour_cos"] = _cos(hour, 24)
        added += ["hour_sin", "hour_cos"]
    else:
        out = out.drop(columns=[c for c in ["hour_sin", "hour_cos"] if c in out.columns])

    # day_of_week
    dow_cfg = getattr(cal_cfg, "day_of_week", None)
    if dow_cfg and bool(getattr(dow_cfg, "enabled", False)):
        enc = str(getattr(dow_cfg, "encoding", "sin_cos")).lower()
        if enc != "sin_cos":
            raise ValueError(f"Unsupported day_of_week encoding: {enc!r} (expected 'sin_cos')")
        out["dow_sin"] = _sin(dow, 7)
        out["dow_cos"] = _cos(dow, 7)
        added += ["dow_sin", "dow_cos"]
    else:
        out = out.drop(columns=[c for c in ["dow_sin", "dow_cos"] if c in out.columns])

    wk_cfg = getattr(cal_cfg, "is_weekend", None)
    hol_cfg = getattr(cal_cfg, "is_holiday", None)

    # is_weekend
    wk_cfg = getattr(cal_cfg, "is_weekend", None)

    # support both dict and pydantic model
    wk_enabled = False
    if isinstance(wk_cfg, dict):
        wk_enabled = bool(wk_cfg.get("enabled", False))
    elif wk_cfg is not None:
        wk_enabled = bool(getattr(wk_cfg, "enabled", False))

    if wk_enabled:
        out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)
        added.append("is_weekend")
    else:
        if "is_weekend" in out.columns:
            out = out.drop(columns=["is_weekend"])

    # is_holiday
    n_holidays = 0
    hol_cfg = getattr(cal_cfg, "is_holiday", None)

    hol_enabled = False
    cal_name = "US"

    if isinstance(hol_cfg, dict):
        hol_enabled = bool(hol_cfg.get("enabled", False))
        cal_name = str(hol_cfg.get("calendar_name", "US"))
    elif hol_cfg is not None:
        hol_enabled = bool(getattr(hol_cfg, "enabled", False))
        cal_name = str(getattr(hol_cfg, "calendar_name", "US"))

    if hol_enabled:
        try:
            import holidays  # type: ignore
            start_year = int(ts.dt.year.min())
            end_year = int(ts.dt.year.max())
            hol = holidays.country_holidays(cal_name, years=range(start_year, end_year + 1))

            is_holiday = ts.dt.date.apply(lambda d: 1 if d in hol else 0).astype(np.int8)
            out["is_holiday"] = is_holiday
            n_holidays = int(is_holiday.sum())
        except Exception:
            out["is_holiday"] = np.zeros(len(out), dtype=np.int8)
            n_holidays = 0

        added.append("is_holiday")
    else:
        if "is_holiday" in out.columns:
            out = out.drop(columns=["is_holiday"])

    stats = TimeFeatureStats(
        rows_in=rows_in,
        rows_out=rows_out,
        inferred_freq=inferred_freq,
        n_holidays=n_holidays,
        features_added=added,
    )
    return out, stats


def build_processed_frames(
    canonical_df: pd.DataFrame,
    data_cfg: DataCfg,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], CleanStats]:
    """
    Canonical (timestamp + zones + ERCOT) -> processed frames:

    Returns:
      - system_df: ['timestamp', 'load_mw']  (ALWAYS)
      - zones_df:  ['timestamp', zones..., 'ERCOT']
      - stats: CleanStats for reporting
    """
    if canonical_df is None or canonical_df.empty:
        raise ValueError("canonical_df is empty. Check ingestion output and raw files.")

    # ----- Config extraction  -----
    ts_col = getattr(data_cfg.data_schema, "timestamp_col", "timestamp")
    freq = getattr(data_cfg.data_schema, "freq", "h")
    tz_note = getattr(data_cfg.data_schema, "timezone", "America/Chicago")  # metadata only

    # Hour convention note
    hour_convention = getattr(data_cfg.data_schema, "hour_convention", "ending")
    hour_note = f"ERCOT Hour {hour_convention.capitalize()} (stored naive local time)"

    target_raw = getattr(data_cfg.target, "column", DEFAULT_SYSTEM_COL)

    missing_policy = "drop"
    if getattr(data_cfg, "missingness", None) is not None:
        missing_policy = getattr(data_cfg.missingness, "policy", "drop")

    save_zones = False
    if getattr(data_cfg, "output", None) is not None:
        save_zones = bool(getattr(data_cfg.output, "save_zones", False))

    # ----- Start processing -----
    df = canonical_df.copy()

    if ts_col not in df.columns:
        raise KeyError(f"Expected timestamp column {ts_col!r} not found. Got columns: {df.columns.tolist()[:20]}")

    if target_raw not in df.columns:
        raise KeyError(f"Expected target column {target_raw!r} not found. Got columns: {df.columns.tolist()[:20]}")

    rows_in = int(len(df))

    # Ensure datetime + naive
    df = _ensure_datetime_naive(df, ts_col=ts_col)

    # Drop bad timestamps
    df = df.dropna(subset=[ts_col]).reset_index(drop=True)
    rows_after_drop_bad_ts = int(len(df))

    # Sort + dedupe first
    df, dup_dropped_1 = _sort_and_dedupe_keep_last(df, ts_col=ts_col)
    rows_after_dedupe = int(len(df))

    # estimate missing hours on the hourly grid (naive)
    missing_hours = _estimate_missing_hours(df, ts_col=ts_col, freq=freq)

    # Apply missing policy on the system target (ERCOT)
    df, missing_target_dropped = _apply_missing_policy_target(
        df, ts_col=ts_col, target_col=target_raw, policy=missing_policy
    )
    rows_after_missing = int(len(df))

    # Dedupe again in case imputation introduced anything unexpected (rare)
    df, dup_dropped_2 = _sort_and_dedupe_keep_last(df, ts_col=ts_col)
    duplicates_dropped = int(dup_dropped_1 + dup_dropped_2)

    # Build system 2-col output
    system_df = df[[ts_col, target_raw]].rename(columns={target_raw: "load_mw"}).copy()
    # Enforce hourly continuity (HYBRID)
    system_df, system_hourly_stats = enforce_hourly_index_hybrid(
        system_df,
        data_cfg=data_cfg,
        ts_col=ts_col,
        value_cols=["load_mw"],
    )

    system_df, tf_stats_system = add_time_features(system_df, data_cfg=data_cfg, ts_col=ts_col)

    # zones output
    zones_df: Optional[pd.DataFrame] = None
    zones_hourly_stats = None
    if save_zones:
        keep = [ts_col] + [c for c in ZONES if c in df.columns] + ([target_raw] if target_raw in df.columns else [])
        zones_df = df[keep].copy()
        zone_cols = [c for c in zones_df.columns if c != ts_col]

        zones_df, zones_hourly_stats = enforce_hourly_index_hybrid(
            zones_df,
            data_cfg=data_cfg,
            ts_col=ts_col,
            value_cols=zone_cols,
        )

        zones_df, tf_stats_zones = add_time_features(zones_df, data_cfg=data_cfg, ts_col=ts_col)

    start_ts = str(system_df[ts_col].min()) if not system_df.empty else ""
    end_ts = str(system_df[ts_col].max()) if not system_df.empty else ""

    stats = CleanStats(
        rows_in=rows_in,
        rows_after_drop_bad_timestamps=rows_after_drop_bad_ts,
        rows_after_dedupe=rows_after_dedupe,
        rows_after_missing_policy=rows_after_missing,

        duplicates_dropped=duplicates_dropped,
        missing_target_dropped=int(missing_target_dropped),

        missing_hours_estimated=int(missing_hours),

        start_timestamp=start_ts,
        end_timestamp=end_ts,
        freq=str(freq),
        timezone_note=f"Stored as naive local time; interpret as {tz_note}",
        missing_policy=str(missing_policy),
        hour_convention_note=hour_note,

        rows_out_system=int(len(system_df)),
        rows_out_zones=int(len(zones_df)) if zones_df is not None else 0,
        save_zones=bool(save_zones),

        hourly_expected_hours_system=system_hourly_stats.expected_hours,
        hourly_missing_inserted_system=system_hourly_stats.missing_hours_inserted,
        hourly_imputed_cells_system=system_hourly_stats.imputed_cells,
        hourly_rows_dropped_after_impute_system=system_hourly_stats.rows_dropped_after_impute,

        hourly_expected_hours_zones=(
            zones_hourly_stats.expected_hours if zones_hourly_stats else 0
        ),
        hourly_missing_inserted_zones=(
            zones_hourly_stats.missing_hours_inserted if zones_hourly_stats else 0
        ),
        hourly_imputed_cells_zones=(
            zones_hourly_stats.imputed_cells if zones_hourly_stats else 0
        ),
        hourly_rows_dropped_after_impute_zones=(
            zones_hourly_stats.rows_dropped_after_impute if zones_hourly_stats else 0
        ),

    )

    return system_df, zones_df, stats