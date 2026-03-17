from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from ..initialization.bootstrap import bootstrap_task, default_task_config
from ..initialization.config import project_runtime_config
from ..initialization.file_io import read_json


def run_demo(max_iterations: int = 20, timeout_seconds: int = 120) -> None:
    project_id = "demo-{}".format(int(time.time()))
    cfg = project_runtime_config(project_id)
    cfg.ensure_dirs()
    task_cfg = default_task_config(task_id="branin-demo-{}".format(int(time.time())))
    task_cfg.max_iterations = max_iterations
    bootstrap_task(task_cfg, runtime_cfg=cfg)

    env = dict(os.environ)
    src_dir = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    worker = subprocess.Popen(
        [sys.executable, "-m", "plat_bo.optimizer.worker", "--project-id", project_id], env=env
    )
    supervisor = subprocess.Popen(
        [sys.executable, "-m", "plat_bo.optimizer.supervisor", "--project-id", project_id], env=env
    )
    start = time.time()
    state_file = cfg.state_dir / "{}_engine_state.json".format(task_cfg.task_id)
    try:
        while True:
            if state_file.exists():
                history = read_json(state_file).get("history", [])
                if len(history) >= max_iterations + 1:
                    break
            if time.time() - start > timeout_seconds:
                raise TimeoutError("branin demo timeout")
            time.sleep(1.0)
    finally:
        worker.terminate()
        supervisor.terminate()
        worker.wait(timeout=10)
        supervisor.wait(timeout=10)

    history = read_json(state_file).get("history", [])
    success = [x for x in history if x.get("success")]
    best = min(success, key=lambda x: x["objective"])
    print("Branin demo finished.")
    print("iterations:", len(history))
    print("best objective:", best["objective"])
    print("best parameters:", best["parameters"])


if __name__ == "__main__":
    run_demo()
