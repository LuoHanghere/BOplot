from __future__ import annotations

import os
import time
from pathlib import Path

from ..initialization.file_io import read_json, write_json


def write_heartbeat(heartbeat_dir: Path, process_name: str) -> None:
    payload = {"pid": os.getpid(), "ts": time.time(), "process": process_name}
    write_json(heartbeat_dir / "{}.json".format(process_name), payload)


def read_heartbeat(heartbeat_dir: Path, process_name: str) -> dict:
    path = heartbeat_dir / "{}.json".format(process_name)
    if not path.exists():
        return {"running": False, "reason": "missing"}
    payload = read_json(path)
    now = time.time()
    running = (now - float(payload.get("ts", 0.0))) <= 3.0
    payload["running"] = running
    return payload

