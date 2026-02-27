from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Iterable, Tuple, Union

import pandas as pd

from src.config.schema import DataCfg

# ERCOT Native load columns expected (considering case-insensitive match after normalization)
RAW_TIMESTAMP_CANDIDATES = {"hour ending", "hour_ending", "hourending"}
ZONE_COLS = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST", "ERCOT"]

@dataclass(frozen=True)
class IngestYearStats:
    year : int
    zip_path: Path
    xlsx_file: str
    chosen_sheet: str
    rows_read: int
    rows_after_basic_clean: int
    dropped_bad_timestamps: int
    dropped_missing_target: int

def _normalize_columns(cols: List[str]) -> List[str]:
    out: List[str] = []
    for c in cols:
        c2 = str(c).strip()
        c2 = " ".join(c2.split())
        out.append(c2)
    return out

def _find_first_xlsx_file(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        files = [file for file in zf.namelist() if file.lower().endswith(".xlsx") and not file.endswith("/")]
        if not files:
            raise FileNotFoundError(f"No xlsx file found in {zip_path}")
        files.sort()
        return files[0]

def _pick_sheet_name(xl: pd.ExcelFile, preferred: str | int | None) -> str | int:
    sheets = xl.sheet_names

    # If user passed an integer (0 = first sheet), respect it.
    if isinstance(preferred, int):
        return preferred

    # Exact match first
    if isinstance(preferred, str) and preferred in sheets:
        return preferred

    # Normalized match (strip/lower)
    if isinstance(preferred, str):
        pref_norm = preferred.strip().lower()
        for s in sheets:
            if s.strip().lower() == pref_norm:
                return s

    # Heuristic: prefer a sheet that contains "native load" or "load"
    ranked = []
    for s in sheets:
        s_norm = s.strip().lower()
        score = 0
        if "native load" in s_norm:
            score += 2
        if "load" in s_norm:
            score += 1
        ranked.append((score, s))
    ranked.sort(reverse=True)

    if ranked and ranked[0][0] > 0:
        return ranked[0][1]

    # Fallback: first sheet
    return sheets[0]


def _read_xlsx_from_zip(zip_path, *, sheet_name):
    xlsx_member = _find_first_xlsx_file(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(xlsx_member) as f:
            xl = pd.ExcelFile(f, engine="openpyxl")
            chosen = _pick_sheet_name(xl, sheet_name)
            df = xl.parse(chosen)
    return df, xlsx_member, chosen

def _detect_timestamp_column(df: pd.DataFrame) -> str:
    """
        Detect the timestamp column in the dataframe
        for the ERCOT dataset, "Hour Ending" is used as the column name
    """
    cols_norm = {str(c).strip().lower().replace("  ", " "): str(c) for c in df.columns}
    if "Hour Ending" in df.columns:
        return "Hour Ending"

    # fallback for no "Hour Ending" and match variant that has timestamps
    for k_norm, original in cols_norm.items():
        k_key = k_norm.lower().replace(" ", "").replace("_", "")
        if k_key in {x.replace(" ", "").replace("_", "") for x in RAW_TIMESTAMP_CANDIDATES}:
            return original

    raise KeyError(
        f"Could not find timestamp column (expected 'Hour Ending' or variant). Columns: {list(df.columns)[:20]}")

def _standardize_native_load_df(
    df: pd.DataFrame,
    *,
    target_col_raw: str,
) -> pd.DataFrame:
    """
    Return canonical wide df:
      timestamp (naive parsed datetime)
      + all available zones (including target_col_raw if present)
    """
    if df.empty:
        raise ValueError(f"Native Load Sheet is empty.")

    df = df.copy()
    df.columns = _normalize_columns(df.columns)

    ts_col = _detect_timestamp_column(df)

    # Ensure target column exists
    if target_col_raw not in df.columns:
        candidates = [c for c in df.columns if c.strip().upper() == target_col_raw.strip().upper()]
        if candidates:
            df = df.rename(columns={candidates[0]: target_col_raw})
        else:
            raise KeyError(f"Target column {target_col_raw} not found. Columns: {df.columns.tolist()[:20]}")
    
    #keep timestamp and zone columns
    keep = [ts_col] + [c for c in ZONE_COLS if c in df.columns]
    df = df[keep]

    # Parse timestamp (handles your '01/01/2025 01:00' and true Excel datetime)
    ts = pd.to_datetime(df[ts_col], errors="coerce")

    df = df.drop(columns=[ts_col])
    df.insert(0, "timestamp", ts)

    #convert numeric columns
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def _find_year_zip(raw_root: Path, year: int) -> Path:
    year_dir = raw_root / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Year directory not found: {year_dir}")

    zips = sorted(year_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip found in: {year_dir}")

    if len(zips) > 1:
        names = [p.name for p in zips]
        raise RuntimeError(
            f"Multiple zip files found for year {year} in {year_dir}. "
            f"Keep exactly 1 zip per year. Found: {names}"
        )

    return zips[0]

def load_native_load_canonical(
    data_cfg: DataCfg,
) -> Tuple[pd.DataFrame, List[IngestYearStats]]:
    """
    Config-driven ingestion: reads yearly ZIPs and returns canonical wide df + stats.

    Expected raw layout:
      {data_cfg.source.raw_dir}/{year}/*.zip
    """
    raw_root = Path(data_cfg.source.raw_dir)
    years = list(data_cfg.source.years)
    sheet_name = getattr(data_cfg.source, "sheet_name", "Sheet1")
    target_raw = getattr(data_cfg.target, "column", "ERCOT")

    all_dfs: List[pd.DataFrame] = []
    all_stats: List[IngestYearStats] = []

    for year in years:
        zip_path = _find_year_zip(raw_root, int(year))
        raw_df, xlsx_file, chosen_sheet = _read_xlsx_from_zip(zip_path, sheet_name=sheet_name)

        rows_read = int(len(raw_df))
        canon = _standardize_native_load_df(raw_df, target_col_raw=target_raw)

        bad_ts = int(canon["timestamp"].isna().sum())
        canon = canon.dropna(subset=["timestamp"])

        dropped_missing_target = int(canon[target_raw].isna().sum()) if target_raw in canon.columns else 0
        canon = canon.dropna(subset=[target_raw])

        canon = canon.sort_values("timestamp").reset_index(drop=True)

        all_dfs.append(canon)
        all_stats.append(
            IngestYearStats(
                year=int(year),
                zip_path=zip_path,
                xlsx_file=xlsx_file,
                chosen_sheet=chosen_sheet,
                rows_read=rows_read,
                rows_after_basic_clean=int(len(canon)),
                dropped_bad_timestamps=bad_ts,
                dropped_missing_target=dropped_missing_target,
            )
        )

    df = pd.concat(all_dfs, axis=0, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, all_stats

























