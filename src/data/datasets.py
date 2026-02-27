from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import load_config
from src.config.schema import AppConfig, DataCfg


DatasetSource = Literal["system", "zones"]

ZONE_COLS: List[str] = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST"]


# -------------------------
# IO helpers
# -------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_date(s: str) -> pd.Timestamp:
    # accepts YYYY-MM-DD
    return pd.to_datetime(s, errors="raise").normalize()


def _slice_df_by_date_range(df: pd.DataFrame, split: Dict[str, str], ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Inclusive start/end by date:
      start <= ts < (end + 1 day)
    """
    start = _parse_date(split["start"])
    end = _parse_date(split["end"])
    end_exclusive = end + pd.Timedelta(days=1)
    m = (df[ts_col] >= start) & (df[ts_col] < end_exclusive)
    return df.loc[m].copy()


def load_splits_artifact(cfg: AppConfig) -> Dict[str, Any]:
    processed_dir = Path(getattr(cfg.data.source, "processed_dir", "data/processed"))
    path = processed_dir / "splits_v0.json"
    if not path.exists():
        raise FileNotFoundError(f"Splits artifact not found: {path}. Run: python scripts/preprocess.py")
    return _read_json(path)


def load_processed_system_df(cfg: AppConfig) -> pd.DataFrame:
    """
    Loads processed 2-col dataset parquet:
      timestamp (naive datetime)
      load_mw (float)
    """
    processed_dir = Path(getattr(cfg.data.source, "processed_dir", "data/processed"))
    out_cfg = getattr(cfg.data, "output", None)
    fname = getattr(out_cfg, "processed_filename", "ercot_hourly_v0.parquet") if out_cfg else "ercot_hourly_v0.parquet"
    path = processed_dir / fname

    if not path.exists():
        raise FileNotFoundError(f"Processed system parquet not found: {path}. Run: python scripts/preprocess.py")

    df = pd.read_parquet(path).copy()
    if "timestamp" not in df.columns or "load_mw" not in df.columns:
        raise ValueError(f"System parquet must contain ['timestamp','load_mw']. Got: {df.columns.tolist()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    df = df.dropna(subset=["load_mw"]).reset_index(drop=True)

    return df


def load_processed_zones_df(cfg: AppConfig) -> pd.DataFrame:
    """
    Loads processed zones dataset parquet:
      timestamp
      zone columns (COAST..WEST)
      ERCOT

    Standardizes:
      - timestamp datetime
      - numeric columns
      - renames ERCOT -> load_mw (so downstream code is consistent)
    """
    processed_dir = Path(getattr(cfg.data.source, "processed_dir", "data/processed"))
    out_cfg = getattr(cfg.data, "output", None)
    fname = getattr(out_cfg, "zones_filename", "ercot_native_load_zones_v0.parquet") if out_cfg else "ercot_native_load_zones_v0.parquet"
    path = processed_dir / fname

    if not path.exists():
        raise FileNotFoundError(
            f"Zones parquet not found: {path}. "
            "Set cfg.data.output.save_zones=true and rerun: python scripts/preprocess.py"
        )

    df = pd.read_parquet(path).copy()
    if "timestamp" not in df.columns:
        raise ValueError(f"Zones parquet must contain 'timestamp'. Got: {df.columns.tolist()}")

    if "ERCOT" not in df.columns:
        raise ValueError(f"Zones parquet must contain 'ERCOT' for system target. Got: {df.columns.tolist()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # numeric conversion
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure target available
    df = df.dropna(subset=["ERCOT"]).reset_index(drop=True)

    # rename target to match system dataset
    df = df.rename(columns={"ERCOT": "load_mw"})
    return df


# -------------------------
# Window specification + index builder
# -------------------------

@dataclass(frozen=True)
class WindowSpec:
    context_length: int
    horizon: int
    stride: int
    allow_partial_windows: bool = False


def window_spec_from_cfg(data_cfg: DataCfg) -> WindowSpec:
    w = getattr(data_cfg, "windows", None)
    if w is None:
        raise ValueError("data.windows missing in config.")
    return WindowSpec(
        context_length=int(getattr(w, "context_length")),
        horizon=int(getattr(w, "horizon")),
        stride=int(getattr(w, "stride", 1)),
        allow_partial_windows=bool(getattr(w, "allow_partial_windows", False)),
    )


def build_window_end_indices(n: int, spec: WindowSpec) -> np.ndarray:
    """
    End index t for each sample:
      X: [t-context_length+1 .. t]
      y: [t+1 .. t+horizon]
    """
    c = int(spec.context_length)
    h = int(spec.horizon)
    s = int(spec.stride)

    if c <= 0 or h <= 0 or s <= 0:
        raise ValueError(f"Invalid WindowSpec: {spec}")

    t_min = c - 1
    t_max = n - h - 1

    if t_max < t_min:
        return np.array([], dtype=np.int64)

    return np.arange(t_min, t_max + 1, s, dtype=np.int64)


# -------------------------
# Dataset
# -------------------------

class TimeWindowDataset(Dataset):
    """
    Generic time-window dataset.

    Returns:
      X: [context_length, feature_dim]
      y: [horizon]

    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        spec: WindowSpec,
        ts_col: str = "timestamp",
        feature_cols: List[str],
        target_col: str = "load_mw",
        return_meta: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if df.empty:
            raise ValueError("Split dataframe is empty; check your split ranges or processed parquet.")

        self.ts_col = ts_col
        self.target_col = target_col
        self.feature_cols = list(feature_cols)
        self.spec = spec
        self.return_meta = return_meta
        self.dtype = dtype

        df = df.sort_values(ts_col).reset_index(drop=True)

        # Validate columns
        needed = [ts_col, target_col] + self.feature_cols
        for c in needed:
            if c not in df.columns:
                raise KeyError(f"Column {c!r} not found. Available: {df.columns.tolist()}")

        self._ts = df[ts_col].to_numpy()
        X = df[self.feature_cols].to_numpy(dtype=np.float32)  # [n, feature_dim]
        y = df[target_col].to_numpy(dtype=np.float32)         # [n]

        self._Xbase = X
        self._ybase = y
        self._n = len(df)

        self._t_ends = build_window_end_indices(self._n, spec)
        if len(self._t_ends) == 0:
            raise ValueError(
                f"No valid windows for n={self._n} with spec={spec}. "
                "Reduce context_length/horizon or check split sizes."
            )

    def __len__(self) -> int:
        return int(len(self._t_ends))

    def __getitem__(self, idx: int):
        t = int(self._t_ends[idx])
        c = int(self.spec.context_length)
        h = int(self.spec.horizon)

        x0 = t - c + 1
        x1 = t + 1
        y0 = t + 1
        y1 = t + 1 + h

        X = torch.as_tensor(self._Xbase[x0:x1, :], dtype=self.dtype)  # [c, feature_dim]
        y = torch.as_tensor(self._ybase[y0:y1], dtype=self.dtype)  # [h]

        if not self.return_meta:
            return X, y

        # helper: numpy.datetime64 -> ISO string (collate-safe)
        def _dt_to_str(v) -> str:
            return str(v)  # e.g. '2024-07-01T01:00:00.000000000'

        meta = {
            "t_end": _dt_to_str(self._ts[t]),
            "x_start": _dt_to_str(self._ts[x0]),
            "x_end": _dt_to_str(self._ts[t]),
            "y_start": _dt_to_str(self._ts[y0]),
            "y_end": _dt_to_str(self._ts[y1 - 1]),

            # for debugging
            "t_end_idx": int(t),
            "x0": int(x0),
            "x1": int(x1),
            "y0": int(y0),
            "y1": int(y1),
        }
        return X, y, meta


# -------------------------
# Factory functions
# -------------------------

def load_source_df(cfg: AppConfig, source: DatasetSource) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Returns:
      df: dataframe with timestamp + features + target
      feature_cols: list of columns used to build X
      target_col: column used to build y (horizon)
    """
    ts_col = "timestamp"
    target_col = "load_mw"

    if source == "system":
        df = load_processed_system_df(cfg)

        if target_col not in df.columns:
            raise ValueError(f"Expected {target_col!r} in system parquet. Columns: {df.columns.tolist()}")

        # Auto-pick: everything except timestamp
        feature_cols = [c for c in df.columns if c != ts_col]

        return df, feature_cols, target_col

    # source == "zones"
    df = load_processed_zones_df(cfg)

    zone_cols = [c for c in ZONE_COLS if c in df.columns]
    if not zone_cols:
        raise ValueError(f"No zone columns found in zones parquet. Expected some of: {ZONE_COLS}")

    # Add any engineered features that exist in the parquet (calendar, holiday, etc.)
    extra_cols = [c for c in df.columns if c not in ([ts_col] + zone_cols)]

    # X = zones + extra engineered features
    feature_cols = zone_cols + extra_cols

    # y = future ERCOT (load_mw). Ensure it's present (your zones parquet includes ERCOT; you may map it to load_mw elsewhere)
    if target_col not in df.columns:
        # If your zones parquet uses 'ERCOT' as the target column, you can map it here:
        if "ERCOT" in df.columns:
            df = df.copy()
            df[target_col] = df["ERCOT"]
        else:
            raise ValueError(
                f"Expected {target_col!r} (or 'ERCOT') in zones parquet for target. "
                f"Columns: {df.columns.tolist()}"
            )

    return df, feature_cols, target_col


def make_datasets(
    cfg: Optional[AppConfig] = None,
    *,
    source: DatasetSource = "system",
    return_meta: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Tuple[TimeWindowDataset, TimeWindowDataset, TimeWindowDataset]:
    """
    Split-safe datasets. Windows are built *within each split only*.
    """
    cfg = cfg or load_config()
    spec = window_spec_from_cfg(cfg.data)
    splits = load_splits_artifact(cfg)

    df, feature_cols, target_col = load_source_df(cfg, source)

    train_df = _slice_df_by_date_range(df, splits["train"])
    val_df = _slice_df_by_date_range(df, splits["val"])
    test_df = _slice_df_by_date_range(df, splits["test"])

    train_ds = TimeWindowDataset(
        train_df, spec=spec, feature_cols=feature_cols, target_col=target_col, return_meta=return_meta, dtype=dtype
    )
    val_ds = TimeWindowDataset(
        val_df, spec=spec, feature_cols=feature_cols, target_col=target_col, return_meta=return_meta, dtype=dtype
    )
    test_ds = TimeWindowDataset(
        test_df, spec=spec, feature_cols=feature_cols, target_col=target_col, return_meta=return_meta, dtype=dtype
    )

    return train_ds, val_ds, test_ds


def make_dataloaders(
    cfg: Optional[AppConfig] = None,
    *,
    source: DatasetSource = "system",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    shuffle_train: bool = True,
    return_meta: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = make_datasets(cfg, source=source, return_meta=return_meta, dtype=dtype)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader