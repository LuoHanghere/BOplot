# Fluent 集成方案（二维翼型 5 控制点）

## 目标定义
- 设计变量：5 个控制点权重 `w1..w5`
- 多目标（最小化形式）：
  - `f1 = -Cl`（等价于最大化升力系数）
  - `f2 = Cd`（最小化阻力系数）
- 平台输出兼容：
  - `objective`: 标量代理目标（当前为 `f1 + 5*f2`）
  - `objective_vector`: `[f1, f2]`

## 现有接入路径
- 问题名：`fluent_airfoil_2d`
- 评估入口：`plat_bo.objective.fluent_airfoil_program`
- 工作流引擎：`plat_bo.objective.fluent_workflow`
- `worker` 已按 `problem` 自动选择评估器，并将 `problem_config` 透传给 Fluent 程序。

## 输入输出协议

### 输入（平台 -> fluent 程序）
```json
{
  "task_id": "airfoil-001",
  "iteration": 12,
  "parameters": [0.50, 0.42, 0.61, 0.47, 0.58],
  "problem_config": {
    "mode": "external",
    "timeout_seconds": 2400,
    "result_json": "result.json",
    "stage_commands": {
      "parameterize": ["python", "-m", "plat_bo.objective.fluent_demo.parametric_model", "--input", "{input_json}", "--workspace", "{workspace}"],
      "mesh": ["python", "-m", "plat_bo.objective.fluent_demo.mesh", "--workspace", "{workspace}"],
      "setup": ["python", "-m", "plat_bo.objective.fluent_demo.setup_case", "--workspace", "{workspace}"],
      "solve": ["python", "-m", "plat_bo.objective.fluent_demo.run_fluent", "--workspace", "{workspace}"],
      "extract": ["python", "-m", "plat_bo.objective.fluent_demo.extract_result", "--workspace", "{workspace}", "--output", "{result_json}"]
    }
  }
}
```

### 输出（fluent 程序 -> 平台）
```json
{
  "objective": -0.81,
  "objective_vector": [-0.95, 0.028],
  "lift_coefficient": 0.95,
  "drag_coefficient": 0.028,
  "success": true,
  "message": "ok"
}
```

## 模式说明
- `mode = mock`：不调用 Fluent，使用代理函数快速联调平台流程。
- `mode = external`：按 `stage_commands` 顺序执行脚本：
  1. parameterize
  2. mesh
  3. setup
  4. solve
  5. extract
- `parameterize` 阶段会产出 `ansys_probe.json`，用于确认 `spaceclaim.exe / fluent.exe / ansysedt.exe` 是否可被调起。
- 示例配置文件：`fluent_airfoil_demo_problem_config.json`，可直接粘贴到 UI 的 `problem_config(JSON)`。

## 下一步建议
- 先在 `mock` 模式完成 MOBO 采集链路联调（qParEGO）。
- 再切到 `external` 模式并接入真实脚本。
- 为 `solve` 阶段加失败重试、超时中断与结果缓存（按参数哈希）。
