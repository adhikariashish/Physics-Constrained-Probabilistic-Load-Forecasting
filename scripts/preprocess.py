from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import sys

# resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.ingest import load_native_load_canonical
from src.data.features import build_processed_frames
from src.data.splits import build_time_splits, save_splits_json

def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    return str(o)  # fallback for any other non-serializable object (safe for stats)

def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")

def _data_quality_report(system_df: pd.DataFrame, ts_col: str = "timestamp", y_col: str = "load_mw") -> Dict[str, Any]:
    rep: Dict[str, Any] = {"rows": int(len(system_df)),
                           "time_min": str(system_df[ts_col].min()) if len(system_df) else "",
                           "time_max": str(system_df[ts_col].max()) if len(system_df) else ""}

    if len(system_df):
        y = system_df[y_col].astype(float)
        rep["load_mw"] = {
            "min": float(y.min()),
            "mean": float(y.mean()),
            "max": float(y.max()),
            "std": float(y.std(ddof=0)),
        }
        dy = y.diff()
        q = dy.quantile([0.95, 0.99, 0.999]).to_dict()
        rep["ramp_delta_mw"] = {str(k): float(v) for k, v in q.items()}
    return rep


def main() -> None:
    cfg = load_config()

    # ---- Ingest ----
    canonical_df, ingest_stats = load_native_load_canonical(cfg.data)

    # ---- Clean + select outputs ----
    system_df, zones_df, clean_stats = build_processed_frames(canonical_df, cfg.data)

    # ---- Resolve output paths from config (with safe defaults) ----
    processed_dir = Path(getattr(cfg.data.source, "processed_dir", "data/processed"))
    out_cfg = getattr(cfg.data, "output", None)

    system_name = getattr(out_cfg, "processed_filename", "ercot_hourly_v0.parquet") if out_cfg else "ercot_hourly_v0.parquet"
    zones_name = getattr(out_cfg, "zones_filename", "ercot_native_load_zones_v0.parquet") if out_cfg else "ercot_native_load_zones_v0.parquet"
    save_zones = bool(getattr(out_cfg, "save_zones", False)) if out_cfg else False

    system_path = processed_dir / system_name
    _write_parquet(system_df, system_path)

    zones_path = None
    if save_zones and zones_df is not None:
        zones_path = processed_dir / zones_name
        _write_parquet(zones_df, zones_path)

    # ---- Splits artifact ----
    splits, counts = build_time_splits(system_df, cfg.data, ts_col="timestamp")
    splits_path = processed_dir / "splits_v0.json"
    save_splits_json(
        splits=splits,
        counts=counts,
        out_path=splits_path,
        extra={
            "processed_file": str(system_path.as_posix()),
            "zones_file": str(zones_path.as_posix()) if zones_path else None,
        },
    )

    # ---- Data quality report ----
    reports_dir = Path("reports/metrics")
    dq = {
        "ingest": [
            {k: (str(v) if isinstance(v, Path) else v) for k, v in s.__dict__.items()}
            for s in ingest_stats
        ],
        "clean_stats": clean_stats.__dict__,
        "system_summary": _data_quality_report(system_df),
    }
    dq_path = reports_dir / "data_quality_v0.json"
    _write_json(dq, dq_path)

    print("✅ Preprocess complete")
    print("System parquet:", system_path.resolve())
    if zones_path:
        print("Zones parquet :", zones_path.resolve())
    print("Splits json   :", splits_path.resolve())
    print("Data quality  :", dq_path.resolve())


if __name__ == "__main__":
    main()