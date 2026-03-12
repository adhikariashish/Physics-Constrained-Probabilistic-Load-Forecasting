from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.config.schema import AppConfig


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


def _deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `upd` into `base` (upd wins).
    """
    out = dict(base)
    for k, v in upd.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _dump_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


@dataclass(frozen=True)
class ConfigPaths:
    root: Path
    data: Path
    model: Path
    train: Path
    eval: Path


def resolve_config_paths(config_dir: str | Path = "configs") -> ConfigPaths:
    root = Path(config_dir).resolve()
    return ConfigPaths(
        root=root,
        data=root / "data.yaml",
        model=root / "model.yaml",
        train=root / "train.yaml",
        eval=root / "eval.yaml",
    )


def load_config(
    *,
    config_dir: str | Path = "configs",
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> AppConfig:
    """
    Load YAML configs from configs/ and return a unified validated AppConfig.

    `overrides` can include partial dict patches like:
      {"train": {"loop": {"epochs": 5}}, "train": {"device": {"name": "cpu"}}}
    """
    paths = resolve_config_paths(config_dir)

    merged: Dict[str, Any] = {}
    for p in [paths.data, paths.model, paths.train, paths.eval]:
        merged = _deep_merge(merged, _read_yaml(p))

    if overrides:
        merged = _deep_merge(merged, overrides)

    if not validate:
        # NOTE: still wrap into AppConfig for dot-access, but skip strict validation.
        return AppConfig.model_construct(**merged)

    return AppConfig.model_validate(merged, by_name=True)


def save_resolved_config(
    *,
    cfg: AppConfig,
    out_path: str | Path,
) -> None:
    """
    Save the resolved config to YAML (for run reproducibility).
    """
    out = Path(out_path)
    # model_dump gives a pure python dict
    data = cfg.model_dump(by_alias=True)
    _dump_yaml(data, out)