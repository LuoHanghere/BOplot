import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src directory to PYTHONPATH so plat_bo can be imported
src_dir = Path(r"g:\LHHHHHHH\Plat\src")
sys.path.insert(0, str(src_dir))
os.environ["PYTHONPATH"] = str(src_dir)

from plat_bo.initialization.bootstrap import bootstrap_task
from plat_bo.initialization.models import TaskConfig
from plat_bo.initialization.config import project_runtime_config
from plat_bo.surrogate.engine import BOEngine

def main():
    project_id = "moo_demo"
    task_id = "dtlz2_3d_test_v2"
    
    # 1. Define bounds for DTLZ2 (d=5 variables, each in [0, 1])
    bounds = [[0.0, 1.0] for _ in range(5)]
    
    # 2. Setup task config with qEHVI strategy
    task_cfg = TaskConfig(
        task_id=task_id,
        problem="dtlz2_3d",
        strategy="base_multi_objective_qehvi",
        bounds=bounds,
        max_iterations=15,
        initial_random_trials=5,
        objective_direction="minimize",
        problem_config={"module": "Mlu.dtlz2_program"},
    )
    
    runtime_cfg = project_runtime_config(project_id)
    runtime_cfg.ensure_dirs()
    
    print(f"Bootstrapping MOO task {task_id} in project {project_id}...")
    bootstrap_task(task_cfg, runtime_cfg=runtime_cfg)
    
    # 3. Start supervisor and worker
    print("Starting Supervisor and Worker...")
    root_dir = Path(r"g:\LHHHHHHH\Plat")
    
    env = dict(os.environ)
    
    worker_proc = subprocess.Popen(
        [sys.executable, "-m", "plat_bo.optimizer.worker", "--project-id", project_id],
        cwd=str(root_dir),
        env=env,
    )
    
    supervisor_proc = subprocess.Popen(
        [sys.executable, "-m", "plat_bo.optimizer.supervisor", "--project-id", project_id],
        cwd=str(root_dir),
        env=env,
    )
    
    # Monitor engine state
    engine = BOEngine(runtime_cfg.state_dir, task_cfg)
    
    try:
        while True:
            time.sleep(2)
            state = engine.get_state(task_id)
            history_len = len(state.history)
            print(f"\rCurrent iterations completed: {history_len} / {task_cfg.max_iterations}", end="")
            
            if history_len >= task_cfg.max_iterations:
                print("\nOptimization finished.")
                break
                
            # Check if processes died
            if worker_proc.poll() is not None or supervisor_proc.poll() is not None:
                print("\nError: Worker or Supervisor died unexpectedly.")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        
    finally:
        print("\nTerminating processes...")
        worker_proc.terminate()
        supervisor_proc.terminate()
        
        # Display best pareto points based on first two objectives (for simplicity)
        state = engine.get_state(task_id)
        if state.history:
            print("\nFinal History Samples (Objective Vectors):")
            for i, item in enumerate(state.history):
                success = item.get("success")
                vec = item.get("objective_vector", [])
                print(f"Iter {i}: Success={success}, Objectives={vec}")

if __name__ == "__main__":
    main()
