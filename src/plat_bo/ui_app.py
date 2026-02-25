from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .bootstrap import bootstrap_task
from .config import RuntimeConfig, project_runtime_config
from .file_io import list_new_json_files, read_json
from .heartbeat import read_heartbeat
from .models import TaskConfig
from .task_config_store import load_task_config


class ProcessManager:
    def __init__(self) -> None:
        self.cfg = RuntimeConfig()
        self.cfg.ensure_dirs()
        self.src_dir = Path(__file__).resolve().parents[1]
        self.root_dir = self.src_dir.parent
        self.worker_proc: subprocess.Popen | None = None
        self.supervisor_proc: subprocess.Popen | None = None
        self.last_error: str = ""
        self.project_id: str | None = None

    def _spawn(self, module_name: str, log_name: str, project_id: str) -> subprocess.Popen:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.src_dir) + (os.pathsep + existing if existing else "")
        project_cfg = project_runtime_config(project_id)
        project_cfg.ensure_dirs()
        log_path = project_cfg.logs_dir / log_name
        log_file = open(log_path, "a", encoding="utf-8")
        return subprocess.Popen(
            [sys.executable, "-m", module_name, "--project-id", project_id],
            env=env,
            cwd=str(self.root_dir),
            stdout=log_file,
            stderr=log_file,
        )

    def _check_alive(self, proc: subprocess.Popen, role: str) -> None:
        time.sleep(0.2)
        if proc.poll() is not None:
            self.last_error = "{} exited immediately. See runtime/state/logs/{}.log".format(
                role, role
            )
            raise RuntimeError(self.last_error)

    def ensure_started(self, project_id: str) -> None:
        self.last_error = ""
        if self.project_id is not None and self.project_id != project_id:
            self.stop_all()
        self.project_id = project_id
        if self.worker_proc is None or self.worker_proc.poll() is not None:
            self.worker_proc = self._spawn("plat_bo.worker", "worker.log", project_id)
            self._check_alive(self.worker_proc, "worker")
        if self.supervisor_proc is None or self.supervisor_proc.poll() is not None:
            self.supervisor_proc = self._spawn("plat_bo.supervisor", "supervisor.log", project_id)
            self._check_alive(self.supervisor_proc, "supervisor")

    def stop_all(self) -> None:
        for proc in [self.worker_proc, self.supervisor_proc]:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.worker_proc = None
        self.supervisor_proc = None
        self.project_id = None

    def restart_all(self) -> None:
        self.stop_all()
        self.ensure_started()

    def process_state(self) -> dict:
        return {
            "worker_pid": None if self.worker_proc is None else self.worker_proc.pid,
            "worker_alive": bool(self.worker_proc is not None and self.worker_proc.poll() is None),
            "supervisor_pid": None if self.supervisor_proc is None else self.supervisor_proc.pid,
            "supervisor_alive": bool(
                self.supervisor_proc is not None and self.supervisor_proc.poll() is None
            ),
            "last_error": self.last_error,
            "project_id": self.project_id,
        }


PROCESS_MANAGER = ProcessManager()


def _parse_bounds(bounds_text: str) -> list[list[float]]:
    pairs = []
    for part in bounds_text.split(";"):
        part = part.strip()
        if not part:
            continue
        lo_str, hi_str = part.split(",")
        pairs.append([float(lo_str.strip()), float(hi_str.strip())])
    return pairs


def _latest_files(directory: Path, limit: int = 10) -> list[str]:
    files = list_new_json_files(directory)
    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    return [p.name for p in files[:limit]]


def _load_state(cfg: RuntimeConfig, task_id: str) -> dict:
    path = cfg.state_dir / "{}_engine_state.json".format(task_id)
    if not path.exists():
        return {}
    return read_json(path)


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Plat BO 控制台</title>
  <style>
    body { font-family: "Segoe UI", sans-serif; margin: 0; background: #f6f8fb; color: #1e293b; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 8px; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .card { background: #fff; border-radius: 12px; padding: 14px; box-shadow: 0 2px 10px rgba(0,0,0,.06); }
    .row { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    input, select, button { width: 100%; padding: 8px; border-radius: 8px; border: 1px solid #cbd5e1; box-sizing: border-box; }
    button { background: #0f766e; color: #fff; border: none; cursor: pointer; }
    pre { background: #0b1220; color: #d1e7ff; padding: 10px; border-radius: 8px; overflow: auto; max-height: 280px; }
    ul { margin: 0; padding-left: 16px; }
    .chart-wrap { height: 320px; }
    canvas { width: 100%; height: 100%; background: #ffffff; border-radius: 10px; border: 1px solid #dbe3ef; }
    .meta { display: flex; gap: 20px; flex-wrap: wrap; margin: 8px 0 12px; color: #334155; }
    .legend { display: flex; gap: 12px; align-items: center; margin-bottom: 8px; color: #334155; }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } .row { grid-template-columns: 1fr; } }
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<div class="wrap">
  <h1>Plat BO 简易前端</h1>
  <p>展示输入输出文件名、是否迭代、优化方案、程序进程、优化问题。<strong>版本: v2-start-button</strong></p>

  <div class="card">
    <h3>任务配置</h3>
    <div class="row">
      <div><label>task_id</label><input id="task_id" value="branin-001"/></div>
      <div><label>project_id</label><input id="project_id" value="test1"/></div>
      <div><label>优化问题</label>
        <select id="problem"><option value="branin">branin</option><option value="mock_quadratic">mock_quadratic</option></select>
      </div>
      <div><label>优化方案</label>
        <select id="strategy"><option value="base_single_task_gp_ei">base_single_task_gp_ei</option></select>
      </div>
      <div><label>边界</label><input id="bounds" value="-5,10;0,15"/></div>
      <div><label>最大迭代</label><input id="max_iter" type="number" value="20"/></div>
      <div><label>初始随机点</label><input id="init_rand" type="number" value="5"/></div>
      <div><label>目标值(可选, 提前停止)</label><input id="obj_target" value=""/></div>
    </div>
    <div style="margin-top:10px;">
      <button onclick="createTask()">创建任务（仅写入首个输入）</button>
    </div>
    <div style="margin-top:10px;" class="row">
      <div><button onclick="createAndStart()">开始计算迭代（创建并启动）</button></div>
      <div><button onclick="stopBackend()" style="background:#b91c1c">停止后端</button></div>
    </div>
    <p id="create_msg"></p>
  </div>

  <div class="grid" style="margin-top:14px;">
    <div class="card"><h3>进程状态</h3><pre id="proc"></pre></div>
    <div class="card"><h3>文件状态</h3><pre id="files"></pre></div>
    <div class="card"><h3>优化进展</h3><pre id="progress"></pre></div>
  </div>

  <div class="card" style="margin-top:14px;">
    <h3>迭代曲线</h3>
    <div class="meta">
      <div>迭代步数: <strong id="iter_now">0</strong>/<span id="iter_max">-</span></div>
      <div>当前值: <strong id="obj_now">-</strong></div>
      <div>历史最优: <strong id="obj_best">-</strong></div>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:#2563eb"></span> objective</span>
      <span><span class="dot" style="background:#ef4444"></span> best-so-far</span>
    </div>
    <div class="chart-wrap"><canvas id="curve"></canvas></div>
  </div>

  <div class="card" style="margin-top:14px;">
    <h3>三维迭代轨迹 (x1, x2, objective)</h3>
    <div id="plot3d" style="height: 520px;"></div>
  </div>
</div>

<script>
function drawCurve(rawSeries, bestSeries) {
  const canvas = document.getElementById("curve");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  if (!rawSeries || rawSeries.length === 0) {
    ctx.fillStyle = "#64748b";
    ctx.font = "14px Segoe UI";
    ctx.fillText("暂无迭代数据", 16, 24);
    return;
  }

  const padding = {l: 46, r: 16, t: 14, b: 30};
  const xs = rawSeries.map(p => p.x);
  const ys = rawSeries.map(p => p.y).concat(bestSeries.map(p => p.y));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }

  const xToPx = x => padding.l + (x - xMin) * (w - padding.l - padding.r) / Math.max(1, (xMax - xMin));
  const yToPx = y => h - padding.b - (y - yMin) * (h - padding.t - padding.b) / (yMax - yMin);

  ctx.strokeStyle = "#dbe3ef";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const gy = padding.t + i * (h - padding.t - padding.b) / 4;
    ctx.beginPath(); ctx.moveTo(padding.l, gy); ctx.lineTo(w - padding.r, gy); ctx.stroke();
  }

  function strokeSeries(series, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((p, i) => {
      const px = xToPx(p.x), py = yToPx(p.y);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
  strokeSeries(rawSeries, "#2563eb");
  strokeSeries(bestSeries, "#ef4444");

  ctx.fillStyle = "#334155";
  ctx.font = "12px Segoe UI";
  ctx.fillText(String(xMin), padding.l, h - 8);
  ctx.fillText(String(xMax), w - padding.r - 12, h - 8);
  ctx.fillText(yMin.toFixed(3), 4, h - padding.b);
  ctx.fillText(yMax.toFixed(3), 4, padding.t + 8);
}

function draw3D(points) {
  const container = document.getElementById("plot3d");
  if (!window.Plotly) {
    container.innerHTML = "<p style='color:#64748b'>Plotly 未加载，无法显示三维图。</p>";
    return;
  }
  if (!points || points.length === 0) {
    container.innerHTML = "<p style='color:#64748b'>暂无三维迭代数据。</p>";
    return;
  }

  const xs = points.map(p => p.x1);
  const ys = points.map(p => p.x2);
  const zs = points.map(p => p.objective);
  const labels = points.map(p => "iter " + p.iteration);

  const trace = {
    type: "scatter3d",
    mode: "lines+markers+text",
    x: xs,
    y: ys,
    z: zs,
    text: labels,
    textposition: "top center",
    marker: {
      size: 5,
      color: points.map(p => p.iteration),
      colorscale: "Viridis",
      colorbar: { title: "iteration" }
    },
    line: { width: 3, color: "#0f766e" },
    hovertemplate: "iter=%{text}<br>x1=%{x:.4f}<br>x2=%{y:.4f}<br>obj=%{z:.6f}<extra></extra>"
  };

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 20 },
    scene: {
      xaxis: { title: "x1" },
      yaxis: { title: "x2" },
      zaxis: { title: "objective" }
    }
  };

  Plotly.newPlot(container, [trace], layout, { responsive: true, displaylogo: false });
}

async function createTask() {
  const body = {
    task_id: document.getElementById("task_id").value,
    project_id: document.getElementById("project_id").value,
    problem: document.getElementById("problem").value,
    strategy: document.getElementById("strategy").value,
    bounds: document.getElementById("bounds").value,
    max_iterations: Number(document.getElementById("max_iter").value),
    initial_random_trials: Number(document.getElementById("init_rand").value),
    objective_target: document.getElementById("obj_target").value.trim() === "" ? null : Number(document.getElementById("obj_target").value),
  };
  const res = await fetch("/api/bootstrap", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
  const data = await res.json();
  document.getElementById("create_msg").innerText = data.message || "ok";
}

function currentTaskPayload() {
  return {
    task_id: document.getElementById("task_id").value,
    project_id: document.getElementById("project_id").value,
    problem: document.getElementById("problem").value,
    strategy: document.getElementById("strategy").value,
    bounds: document.getElementById("bounds").value,
    max_iterations: Number(document.getElementById("max_iter").value),
    initial_random_trials: Number(document.getElementById("init_rand").value),
    objective_target: document.getElementById("obj_target").value.trim() === "" ? null : Number(document.getElementById("obj_target").value),
  };
}

async function createAndStart() {
  const res = await fetch("/api/task/start", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(currentTaskPayload())
  });
  const data = await res.json();
  document.getElementById("create_msg").innerText = data.message || "started";
  await refresh();
}

async function startBackend() {
  const res = await fetch("/api/process/start", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({project_id: document.getElementById("project_id").value})
  });
  const data = await res.json();
  document.getElementById("create_msg").innerText = data.message || "started";
}

async function stopBackend() {
  const res = await fetch("/api/process/stop", {method:"POST"});
  const data = await res.json();
  document.getElementById("create_msg").innerText = data.message || "stopped";
}

async function refresh() {
  const taskId = document.getElementById("task_id").value;
  const projectId = document.getElementById("project_id").value;
  const res = await fetch("/api/status?task_id=" + encodeURIComponent(taskId) + "&project_id=" + encodeURIComponent(projectId));
  const data = await res.json();
  document.getElementById("proc").innerText = JSON.stringify(data.process, null, 2);
  document.getElementById("files").innerText = JSON.stringify(data.files, null, 2);
  document.getElementById("progress").innerText = JSON.stringify(data.progress, null, 2);
  document.getElementById("iter_now").innerText = String(data.progress.current_iteration ?? 0);
  document.getElementById("iter_max").innerText = String(data.progress.max_iterations ?? "-");
  document.getElementById("obj_now").innerText = data.progress.current_objective == null ? "-" : Number(data.progress.current_objective).toFixed(6);
  document.getElementById("obj_best").innerText = data.progress.best_objective == null ? "-" : Number(data.progress.best_objective).toFixed(6);
  drawCurve(data.progress.series || [], data.progress.best_series || []);
  draw3D(data.progress.points3d || []);
}
setInterval(refresh, 2000);
refresh();
window.addEventListener("resize", refresh);
</script>
</body>
</html>
"""

UI_VERSION = "v2-start-button"


class UIHandler(BaseHTTPRequestHandler):
    cfg = RuntimeConfig()

    def _write_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._write_html(HTML)
            return
        if parsed.path == "/api/status":
            q = parse_qs(parsed.query)
            task_id = q.get("task_id", ["branin-001"])[0]
            project_id = q.get("project_id", ["test1"])[0]
            self._write_json(self._build_status(task_id, project_id))
            return
        if parsed.path == "/api/version":
            self._write_json({"version": UI_VERSION})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/process/start":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
            project_id = str(payload.get("project_id", "test1"))
            try:
                PROCESS_MANAGER.ensure_started(project_id)
                self._write_json({"success": True, "message": "backend started"})
            except Exception as exc:  # noqa: BLE001
                self._write_json({"success": False, "message": str(exc)}, status=500)
            return
        if parsed.path == "/api/process/stop":
            PROCESS_MANAGER.stop_all()
            self._write_json({"success": True, "message": "backend stopped"})
            return
        if parsed.path == "/api/task/start":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
            try:
                bounds = _parse_bounds(str(payload.get("bounds", "")))
                config = TaskConfig(
                    task_id=str(payload["task_id"]),
                    problem=str(payload["problem"]),
                    strategy=str(payload["strategy"]),
                    bounds=bounds,
                    max_iterations=int(payload.get("max_iterations", 20)),
                    initial_random_trials=int(payload.get("initial_random_trials", 5)),
                    objective_direction="minimize",
                    objective_target=(
                        None
                        if payload.get("objective_target") is None
                        else float(payload.get("objective_target"))
                    ),
                )
                project_id = str(payload.get("project_id", "test1"))
                runtime_cfg = project_runtime_config(project_id)
                bootstrap_task(config, runtime_cfg=runtime_cfg)
                try:
                    PROCESS_MANAGER.ensure_started(project_id)
                except Exception as exc:  # noqa: BLE001
                    self._write_json({"success": False, "message": str(exc)}, status=500)
                    return
                self._write_json(
                    {
                        "success": True,
                        "message": "task created and iteration started",
                        "backend": PROCESS_MANAGER.process_state(),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._write_json({"success": False, "message": str(exc)}, status=400)
            return
        if parsed.path != "/api/bootstrap":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw or "{}")
        try:
            bounds = _parse_bounds(str(payload.get("bounds", "")))
            config = TaskConfig(
                task_id=str(payload["task_id"]),
                problem=str(payload["problem"]),
                strategy=str(payload["strategy"]),
                bounds=bounds,
                max_iterations=int(payload.get("max_iterations", 20)),
                initial_random_trials=int(payload.get("initial_random_trials", 5)),
                objective_direction="minimize",
                objective_target=(
                    None if payload.get("objective_target") is None else float(payload.get("objective_target"))
                ),
            )
            project_id = str(payload.get("project_id", "test1"))
            runtime_cfg = project_runtime_config(project_id)
            bootstrap_task(config, runtime_cfg=runtime_cfg)
            self._write_json({"success": True, "message": "task bootstrapped"})
        except Exception as exc:  # noqa: BLE001
            self._write_json({"success": False, "message": str(exc)}, status=400)

    def _build_status(self, task_id: str, project_id: str) -> dict:
        cfg = project_runtime_config(project_id)
        cfg.ensure_dirs()
        worker = read_heartbeat(cfg.heartbeat_dir, "worker")
        supervisor = read_heartbeat(cfg.heartbeat_dir, "supervisor")
        process = {
            "worker": worker,
            "supervisor": supervisor,
            "iterating": bool(worker.get("running") and supervisor.get("running")),
            "manager": PROCESS_MANAGER.process_state(),
        }
        files = {
            "inbox": _latest_files(cfg.inbox_dir),
            "outbox": _latest_files(cfg.outbox_dir),
            "processed": _latest_files(cfg.processed_dir),
        }
        progress = {"task_id": task_id, "history_count": 0}
        try:
            task_cfg = load_task_config(cfg.task_config_dir, task_id)
            state = _load_state(cfg, task_id)
            history = state.get("history", [])
            progress["task_config"] = task_cfg.to_dict()
            progress["history_count"] = len(history)
            progress["max_iterations"] = int(task_cfg.max_iterations)
            success = [x for x in history if x.get("success")]
            series = []
            best_series = []
            points3d = []
            best_val = None
            for item in success:
                x = int(item["iteration"])
                y = float(item["objective"])
                series.append({"x": x, "y": y})
                params = item.get("parameters", [])
                if isinstance(params, list) and len(params) >= 2:
                    points3d.append(
                        {
                            "iteration": x,
                            "x1": float(params[0]),
                            "x2": float(params[1]),
                            "objective": y,
                        }
                    )
                if best_val is None or y < best_val:
                    best_val = y
                best_series.append({"x": x, "y": float(best_val)})
            progress["series"] = series
            progress["best_series"] = best_series
            progress["points3d"] = points3d
            if success:
                best = min(success, key=lambda x: x["objective"])
                progress["best"] = best
                progress["best_objective"] = float(best["objective"])
            if history:
                progress["latest"] = history[-1]
                progress["current_iteration"] = int(history[-1]["iteration"])
                progress["current_objective"] = float(history[-1]["objective"])
        except FileNotFoundError:
            progress["task_config"] = None
        progress["server_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return {"process": process, "files": files, "progress": progress}


def run_ui(host: str = "127.0.0.1", port: int = 8765) -> None:
    cfg = RuntimeConfig()
    cfg.ensure_dirs()
    server = ThreadingHTTPServer((host, port), UIHandler)
    print("UI running at http://{}:{}/".format(host, port))
    try:
        server.serve_forever()
    finally:
        PROCESS_MANAGER.stop_all()


if __name__ == "__main__":
    run_ui()
