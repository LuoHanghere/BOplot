# Plat BO 科研优化平台

`Plat BO` 是一个面向科研场景的贝叶斯优化平台原型，强调：
- 任务与数据文件解耦（输入/输出文件协议）
- 优化引擎与仿真程序解耦（子进程评估器）
- 可视化监控与参数配置一体化（内置 Web UI）
- 单目标与多目标统一框架（同一任务调度流程）

它适合用于“昂贵黑盒函数优化”问题，例如 CFD、结构、材料、控制参数调优等。

## 1. 平台定位与设计思想

平台采用“文件队列 + 双进程调度 + 代理模型建议”的结构：
- `worker` 负责消费输入参数并调用外部程序计算目标值
- `supervisor` 负责消费计算结果、更新 BO 状态、生成下一轮参数
- 状态与历史全部落盘，便于追溯、复现、断点恢复和离线分析

与“把优化逻辑写死在脚本里”相比，这个框架更适合科研迭代：
- 能快速接入新的求解器程序
- 能替换优化策略组件
- 能在不改仿真器的情况下升级优化能力

## 2. 架构总览

核心链路如下：

1. UI 或 bootstrap 写入首个 `*_input.json` 到 `inbox`
2. `worker` 读取输入，调用 evaluator 运行外部程序，写 `*_output.json` 到 `outbox`
3. `supervisor` 读取输出，更新 `BOEngine` 历史状态，判断终止条件
4. 若未终止，`supervisor` 生成下一轮输入并写回 `inbox`
5. 全过程记录到 `state/iterations`、`state/tasks`、`state/*engine_state.json`

终止条件：
- 达到 `max_iterations`
- 达到目标阈值 `objective_target`

## 3. `src` 模块说明

代码根目录：`g:\LHHHHHHH\Plat\src\plat_bo`

### 3.1 `initialization`（初始化与配置）
- `config.py`：定义项目级运行目录结构，支持 `project_id` 隔离。
- `models.py`：定义 `TaskConfig / TrialInput / TrialOutput` 数据模型。
- `bootstrap.py`：任务启动入口，完成配置校验、存储、首点生成。
- `validator.py`：输入与任务配置校验。
- `task_config_store.py`：任务配置读写。

### 3.2 `optimizer`（调度与平台进程）
- `worker.py`：从 `inbox` 取输入，调用 evaluator，产出 `outbox`。
- `supervisor.py`：从 `outbox` 取结果，更新引擎，生成下一轮输入。
- `heartbeat.py`：维护 `worker/supervisor` 心跳。
- `ui_app.py`：内置 HTTP UI，支持建任务、启动后端、查看进度、曲线、3D 轨迹、策略组合。
- `run_branin_demo.py`：端到端演示入口。

### 3.3 `surrogate`（BO 状态与建议逻辑）
- `engine.py`：平台核心状态机。
- 早期阶段用 Sobol 采样初始化；
- 后续阶段根据组件配置走 BoTorch 或 TuRBO skeleton；
- 状态保存到 `*_engine_state.json`，含历史与 turbo 状态。

### 3.4 `acquisition`（策略组件化）
- `strategy_config.py`：定义策略预设、组件兼容矩阵与校验逻辑。
- `botorch_acq.py`：实现单目标（EI/UCB）与多目标（qEHVI）建议。
- `turbo_skeleton.py`：TuRBO 骨架实现入口。

### 3.5 `objective`（目标评估层）
- `evaluator.py`：统一评估接口，支持 mock 与子进程评估。
- `branin_program.py`：示例目标程序。
- `su2_airfoil_program.py`：SU2 场景程序集成入口。
- `visualize_3d.py`：离线三维轨迹可视化。

## 4. 运行目录（按项目隔离）

运行时目录位于：`g:\LHHHHHHH\Plat\project\<project_id>`

- `inbox`：待计算输入
- `outbox`：计算输出
- `processed`：归档文件
- `state/tasks`：任务配置
- `state/heartbeat`：进程心跳
- `state/iterations`：逐轮迭代记录
- `state/logs`：`worker/supervisor` 日志
- `state/*_engine_state.json`：BO 状态持久化

## 5. 快速开始

### 5.1 安装

```powershell
pip install -e .
```

### 5.2 启动平台 UI

```powershell
$env:PYTHONPATH='src'
python -m plat_bo
```

浏览器访问：`http://127.0.0.1:8765/`

### 5.3 通过命令行初始化任务（可选）

```powershell
$env:PYTHONPATH='src'
python -m plat_bo.initialization.bootstrap --project-id test1 --task-id branin-001 --max-iterations 20
```

### 5.4 一键 demo

```powershell
$env:PYTHONPATH='src'
python -m plat_bo.optimizer.run_branin_demo
```

## 6. 输入输出协议（核心接口）

输入 `TrialInput` 示例：

```json
{
  "task_id": "branin-001",
  "iteration": 0,
  "parameters": [1.0, 2.0]
}
```

输出 `TrialOutput` 示例：

```json
{
  "task_id": "branin-001",
  "iteration": 0,
  "parameters": [1.0, 2.0],
  "objective": 15.3,
  "success": true,
  "message": "ok",
  "cost_seconds": 0.012,
  "objective_vector": null
}
```

## 7. 当前能力摘要

- 支持单目标 BO（`single_task_gp + EI/UCB`）
- 支持多目标入口（`multi_objective_gp + qEHVI`）
- 支持策略组件化组合与兼容性约束
- 支持项目隔离、多任务并行管理（按目录）
- 支持基础监控（心跳、进度、最优值、迭代曲线、3D 轨迹）
- 支持外部程序子进程调用，便于接 SU2 等科研程序

## 8. 当前不足（基于现有 `src` 实现）

以下问题是平台进一步工程化前需要重点解决的：

1. 稳健性与容错能力仍偏基础  
   - 当前以轮询文件为主，异常恢复、重试策略、幂等控制较轻量。  
   - 对异常退出、脏文件、部分失败场景的自动修复策略还不系统。

2. 任务调度能力偏单机  
   - `worker/supervisor` 主要面向单机进程模式。  
   - 尚未形成面向 HPC/集群的队列调度、资源编排、配额管理能力。

3. 并发与一致性边界未完全产品化  
   - 通过文件系统实现协作，适合原型与中小规模实验。  
   - 在高吞吐、多写者场景下，需要更严格的锁、事务或消息队列机制。

4. 策略生态仍处于“可扩展骨架”阶段  
   - 已有多种策略入口，但高级策略（完整 TuRBO、多保真、约束 BO 等）尚未全面落地。  
   - 自动策略选择与元学习能力尚未建立。

5. 可观测性与实验管理仍需增强  
   - 目前主要依赖 JSON 文件与 UI 面板。  
   - 缺少系统化实验追踪（版本、参数、工况、结果）与报表生成能力。

6. 测试与质量保障需要继续补强  
   - 作为科研平台，接口长期演进快。  
   - 需要更系统的单元测试、集成测试与回归基准保障稳定升级。

## 9. 后续发展方向（建议路线图）

### 9.1 工程化阶段（近期）
- 引入更完善的任务状态机（待执行/执行中/失败重试/终止）。
- 增强异常恢复：断点续跑、失败重放、输入去重与幂等控制。
- 完善日志分层与指标埋点，统一健康检查接口。

### 9.2 扩展阶段（中期）
- 对接消息队列或数据库，降低对文件轮询的耦合。
- 支持多 worker 扩展与远程调度（本地 + 集群混合模式）。
- 引入实验管理模块，打通任务配置、代码版本、结果产出闭环。

### 9.3 智能化阶段（中长期）
- 扩展高级 BO：约束 BO、多保真 BO、批量并行 BO、鲁棒 BO。
- 引入策略自动选择与 AutoBO 能力（依据任务特征推荐策略）。
- 构建“优化建议 + 机理分析”联合框架，提升科研解释性。

## 10. 适用场景与建议

适用：
- 单次评估昂贵、变量维度中低、希望自动迭代优化的科研任务。
- 需要把优化逻辑与仿真程序解耦并持续迭代的平台型项目。

建议：
- 先用 `branin` 或 `mock_quadratic` 完成流程验证；
- 再接入真实程序（如 SU2），逐步增加维度与约束；
- 通过项目隔离目录组织不同实验主题，便于可复现实验管理。
