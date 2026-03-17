from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_workspace(workspace: str | Path, base_dir: Path | None = None) -> Path:
    raw = Path(workspace)
    if base_dir is None:
        path = raw.resolve()
    else:
        base = Path(base_dir).resolve()
        if raw.is_absolute():
            path = raw.resolve()
        else:
            path = (base / raw).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parameters(data: dict[str, Any], count: int = 5) -> list[float]:
    params = data.get("parameters", [])
    if not isinstance(params, list) or len(params) != count:
        raise ValueError("parameters length must be {}".format(count))
    return [float(x) for x in params]


def read_env_config(root: Path) -> dict[str, str]:
    path = root / "env_config.json"
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(k, str)}
