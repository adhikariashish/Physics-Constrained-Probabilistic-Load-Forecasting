from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_json(path: Path, data: Dict[str, Any], *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent))