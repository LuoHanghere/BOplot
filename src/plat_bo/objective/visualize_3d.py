from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ..initialization.config import project_runtime_config
from ..initialization.task_config_store import load_task_config


def _load_engine_history(project_id: str, task_id: str) -> list[dict[str, Any]]:
    cfg = project_runtime_config(project_id)
    state_file = cfg.state_dir / "{}_engine_state.json".format(task_id)
    if not state_file.exists():
        raise FileNotFoundError("state file not found: {}".format(state_file))
    with state_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("history", []))


def _discover_tasks(project_id: str) -> list[str]:
    cfg = project_runtime_config(project_id)
    task_dir = cfg.task_config_dir
    tasks: list[str] = []
    if task_dir.exists():
        tasks = sorted(p.stem for p in task_dir.glob("*.json") if p.is_file())
    if tasks:
        return tasks
    state_dir = cfg.state_dir
    if not state_dir.exists():
        return []
    return sorted(
        p.name[: -len("_engine_state.json")]
        for p in state_dir.glob("*_engine_state.json")
        if p.is_file()
    )


def _extract_points(history: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[int]]:
    x1: list[float] = []
    x2: list[float] = []
    obj: list[float] = []
    iters: list[int] = []
    for item in history:
        if not bool(item.get("success", False)):
            continue
        params = item.get("parameters", [])
        if not isinstance(params, list) or len(params) < 2:
            continue
        x1.append(float(params[0]))
        x2.append(float(params[1]))
        obj.append(float(item["objective"]))
        iters.append(int(item["iteration"]))
    return x1, x2, obj, iters


def _branin_value(x1: float, x2: float) -> float:
    a = 1.0
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1.0 - t) * math.cos(x1) + s


def _build_branin_surface(
    bounds: list[list[float]], grid_size: int = 60
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    x1_min, x1_max = float(bounds[0][0]), float(bounds[0][1])
    x2_min, x2_max = float(bounds[1][0]), float(bounds[1][1])
    try:
        import numpy as np

        x1 = np.linspace(x1_min, x1_max, grid_size)
        x2 = np.linspace(x2_min, x2_max, grid_size)
        x1g, x2g = np.meshgrid(x1, x2)
        a = 1.0
        b = 5.1 / (4.0 * math.pi * math.pi)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)
        z = a * (x2g - b * x1g * x1g + c * x1g - r) ** 2 + s * (1.0 - t) * np.cos(x1g) + s
        return x1g.tolist(), x2g.tolist(), z.tolist()
    except Exception:  # noqa: BLE001
        x1_step = (x1_max - x1_min) / float(grid_size - 1)
        x2_step = (x2_max - x2_min) / float(grid_size - 1)
        x1g: list[list[float]] = []
        x2g: list[list[float]] = []
        zg: list[list[float]] = []
        for i in range(grid_size):
            row_x1: list[float] = []
            row_x2: list[float] = []
            row_z: list[float] = []
            x2 = x2_min + i * x2_step
            for j in range(grid_size):
                x1 = x1_min + j * x1_step
                row_x1.append(x1)
                row_x2.append(x2)
                row_z.append(_branin_value(x1, x2))
            x1g.append(row_x1)
            x2g.append(row_x2)
            zg.append(row_z)
        return x1g, x2g, zg


def _plot_plotly(
    x1: list[float],
    x2: list[float],
    obj: list[float],
    iters: list[int],
    title: str,
    output_html: Path | None,
    show: bool,
    surface: tuple[list[list[float]], list[list[float]], list[list[float]]] | None,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:  # noqa: BLE001
        return False

    traces: list[Any] = []
    if surface is not None:
        sx, sy, sz = surface
        traces.append(
            go.Surface(
                x=sx,
                y=sy,
                z=sz,
                opacity=0.6,
                colorscale="Viridis",
                showscale=False,
            )
        )
    traces.append(
        go.Scatter3d(
        x=x1,
        y=x2,
        z=obj,
        mode="markers+text",
        text=["iter {}".format(i) for i in iters],
        textposition="top center",
        marker=dict(size=5, color=iters, colorscale="Viridis", colorbar=dict(title="iter")),
        hovertemplate="iter=%{text}<br>x1=%{x:.4f}<br>x2=%{y:.4f}<br>obj=%{z:.6f}<extra></extra>",
        )
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="objective"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    if output_html is not None:
        fig.write_html(str(output_html), include_plotlyjs="cdn")
        print("saved html:", output_html)
    if show:
        fig.show()
    return True


def _plot_matplotlib(
    x1: list[float],
    x2: list[float],
    obj: list[float],
    iters: list[int],
    title: str,
    output_png: Path | None,
    show: bool,
    surface: tuple[list[list[float]], list[list[float]], list[list[float]]] | None,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    if surface is not None:
        sx, sy, sz = surface
        try:
            import numpy as np

            ax.plot_surface(
                np.asarray(sx), np.asarray(sy), np.asarray(sz), cmap="viridis", alpha=0.6, linewidth=0
            )
        except Exception:  # noqa: BLE001
            pass
    scatter = ax.scatter(x1, x2, obj, c=iters, cmap="viridis", s=35)
    for xi, yi, zi, it in zip(x1, x2, obj, iters):
        ax.text(xi, yi, zi, str(it), fontsize=8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("objective")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, pad=0.1, label="iteration")
    if output_png is not None:
        fig.savefig(str(output_png), dpi=150, bbox_inches="tight")
        print("saved png:", output_png)
    if show:
        plt.show()
    plt.close(fig)


def _choose_task_interactive(tasks: list[str]) -> str:
    print("available tasks:")
    for idx, name in enumerate(tasks):
        print("  [{}] {}".format(idx, name))
    while True:
        s = input("select task index: ").strip()
        if not s:
            continue
        try:
            i = int(s)
        except ValueError:
            print("please input integer index")
            continue
        if 0 <= i < len(tasks):
            return tasks[i]
        print("index out of range")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone 3D visualization for BO iterations.")
    parser.add_argument("--project-id", required=True, help="e.g. test1")
    parser.add_argument("--task-id", default=None, help="if omitted, choose interactively")
    parser.add_argument("--list-tasks", action="store_true", help="list tasks and exit")
    parser.add_argument("--no-show", action="store_true", help="do not open interactive window")
    parser.add_argument("--output-html", default=None, help="write plotly html")
    parser.add_argument("--output-png", default=None, help="write matplotlib png")
    args = parser.parse_args()

    tasks = _discover_tasks(args.project_id)
    if args.list_tasks:
        if not tasks:
            cfg = project_runtime_config(args.project_id)
            print(
                "no tasks found under {} or state files under {}".format(
                    cfg.task_config_dir, cfg.state_dir
                )
            )
            return
        for t in tasks:
            print(t)
        return

    if args.task_id:
        task_id = args.task_id
    else:
        if not tasks:
            cfg = project_runtime_config(args.project_id)
            raise FileNotFoundError(
                "no tasks found under {} or state files under {}".format(
                    cfg.task_config_dir, cfg.state_dir
                )
            )
        task_id = _choose_task_interactive(tasks)

    history = _load_engine_history(args.project_id, task_id)
    x1, x2, obj, iters = _extract_points(history)
    if not x1:
        raise RuntimeError("no valid 2D points found in history for task {}".format(task_id))

    surface = None
    try:
        cfg = project_runtime_config(args.project_id)
        task_cfg = load_task_config(cfg.task_config_dir, task_id)
        if task_cfg.problem == "branin" and len(task_cfg.bounds) >= 2:
            surface = _build_branin_surface(task_cfg.bounds)
    except Exception:  # noqa: BLE001
        surface = None

    title = "project={} task={} iterations={}".format(args.project_id, task_id, len(x1))
    show = not args.no_show
    output_html = Path(args.output_html) if args.output_html else None
    output_png = Path(args.output_png) if args.output_png else None

    used_plotly = _plot_plotly(x1, x2, obj, iters, title, output_html, show, surface)
    if not used_plotly:
        _plot_matplotlib(x1, x2, obj, iters, title, output_png, show, surface)


if __name__ == "__main__":
    main()
