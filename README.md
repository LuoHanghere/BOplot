# Plat BO Framework

当前版本包含：
- Base Route 正式版引擎：`BoTorch SingleTaskGP + ExpectedImprovement`
- 文件驱动迭代：监听 `inbox`、产出 `outbox`、监督器触发下一轮输入
- Branin 独立程序评估：通过子进程调用 `plat_bo.problems.branin_program`
- 简易前端（Streamlit）：展示输入输出文件名、是否迭代、优化方案、程序进程、优化问题与进展

## 1. 环境安装

```powershell
pip install -e .
```

## 2. 目录说明

运行时目录：
按项目隔离存储：
- `src/project/<project_id>/inbox`：输入文件
- `src/project/<project_id>/outbox`：输出文件
- `src/project/<project_id>/processed`：归档文件
- `src/project/<project_id>/state`：引擎状态
- `src/project/<project_id>/state/tasks`：任务配置
- `src/project/<project_id>/state/heartbeat`：worker/supervisor 心跳
- `src/project/<project_id>/state/iterations`：每步迭代记录（输入/结果/下一步参数）

## 3. Branin 优化测试（端到端）

### 3.1 启动首个任务输入

```powershell
$env:PYTHONPATH='src'
python -m plat_bo.bootstrap --project-id test1 --task-id branin-001 --max-iterations 20
```

### 3.2 启动平台（自动拉起后端）

```powershell
$env:PYTHONPATH='src'
python -m plat_bo
```

该命令会启动前端。点击页面“开始计算迭代（创建并启动）”后，才会拉起 `worker + supervisor`。
`worker` 会调用独立程序 `plat_bo.problems.branin_program` 完成函数计算。

### 3.3 一键 demo（自动拉起 worker+supervisor 并等待完成）

```powershell
$env:PYTHONPATH='src'
python -m plat_bo.run_branin_demo
```

## 4. 前端界面（简易版）

```powershell
$env:PYTHONPATH='src'
python -m plat_bo
```

页面主要能力：
- 展示 `inbox/outbox/processed` 文件名
- 展示是否正在迭代（基于 worker/supervisor 心跳）
- 选择优化方案（当前为 `base_single_task_gp_ei`）
- 配置优化问题（当前内置 Branin）
- 显示当前最优目标值与参数

启动后浏览器访问：`http://127.0.0.1:8765/`

## 5. 输入输出格式

输入：
```json
{
  "task_id": "branin-001",
  "iteration": 0,
  "parameters": [1.0, 2.0]
}
```

## 6. 独立三维可视化程序（检验用）

终端选择 `test1` 项目的任务并绘图：

```powershell
$env:PYTHONPATH='src'
python -m plat_bo.visualize_3d --project-id test1
```

直接指定任务绘图：

```powershell
$env:PYTHONPATH='src'
python -m plat_bo.visualize_3d --project-id test1 --task-id branin-1
```

输出：
```json
{
  "task_id": "branin-001",
  "iteration": 0,
  "parameters": [1.0, 2.0],
  "objective": 15.3,
  "success": true,
  "message": "ok",
  "cost_seconds": 0.012
}
```
