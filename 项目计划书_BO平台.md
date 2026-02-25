# BO实验指导平台项目计划书（基于 `bo优化设计方案.md`）

## 1. 项目目标与范围

### 1.1 总目标
- 建立一个可扩展的贝叶斯优化（BO）实验指导平台基础框架。
- 当前阶段优先实现 Base Route（黑盒连续优化）可运行闭环。
- 通过文件夹驱动实现输入检测、格式校验、结果输出、下一轮仿真触发控制。

### 1.2 本期范围（MVP）
- 输入文件监听与格式校验。
- 调用评估器（当前为可替换的 `MockEvaluator`）执行一次“仿真”。
- 输出标准结果文件。
- 监督器读取结果并更新优化引擎状态，自动生成下一轮输入文件。
- 状态持久化（history）与最大迭代控制。
- 简易前端页面：展示输入输出文件名、是否迭代、优化方案、程序进程、优化问题。
- Branin 函数独立程序测试链路：外部程序计算目标值，平台负责迭代控制。

### 1.3 后续扩展范围（不在本期实现）
- BoTorch `SingleTaskGP + EI` 正式接入。
- cEI（约束）/ EHVI（多目标）分支。
- MultiTaskGP（多保真）分支。
- VAE/SSM 隐空间分支。

## 2. 技术路线与模块映射

### 2.1 技术栈
- 语言：Python 3.10+
- 依赖策略：MVP阶段尽量使用标准库，降低部署复杂度。
- 后续核心依赖：PyTorch、BoTorch、GPyTorch、SciPy（在 Phase 2/3 接入）。

### 2.2 架构分层
- `Worker`：监听输入目录，校验输入，调用 `Evaluator`，产出输出文件。
- `Supervisor`：监听输出目录，读取并更新 `BOEngine`，产出下一轮输入文件。
- `BOEngine`：维护状态与候选点生成（当前为 BoTorch `SingleTaskGP + EI`）。
- `Evaluator`：对接外部仿真/实验执行系统（支持独立子程序调用）。
- `Validator`：输入格式和字段合法性校验。
- `FileIO`：统一 JSON 读写与文件发现。
- `UI`：任务配置、进程心跳、迭代状态与结果展示。

## 3. 目录与交付件

## 3.1 目录结构
```text
Plat/
  pyproject.toml
  README.md
  项目计划书_BO平台.md
  src/plat_bo/
    __init__.py
    config.py
    models.py
    validator.py
    file_io.py
    evaluator.py
    engine.py
    worker.py
    supervisor.py
    bootstrap.py
```

### 3.2 核心交付件
- `src/plat_bo/worker.py`：输入检测、校验、产出结果。
- `src/plat_bo/supervisor.py`：结果检测、状态更新、下一轮触发。
- `src/plat_bo/engine.py`：BoTorch `SingleTaskGP + EI` 优化引擎。
- `src/plat_bo/evaluator.py`：评估抽象接口与 mock 实现。
- `src/plat_bo/problems/branin_program.py`：Branin 独立程序。
- `src/plat_bo/ui_app.py`：简易前端可视化界面。
- `README.md`：运行说明。

## 4. 输入输出规范（第一版）

### 4.1 输入文件（放置于 `runtime/inbox`）
```json
{
  "task_id": "demo-001",
  "iteration": 0,
  "parameters": [0.3, 0.7]
}
```

### 4.2 输入校验规则
- 必填字段：`task_id`、`iteration`、`parameters`。
- `task_id`：非空字符串。
- `iteration`：非负整数。
- `parameters`：非空数值列表（int/float）。

### 4.3 输出文件（生成于 `runtime/outbox`）
```json
{
  "task_id": "demo-001",
  "iteration": 0,
  "objective": 0.08,
  "success": true,
  "message": "ok",
  "cost_seconds": 0.00012
}
```

### 4.4 异常输出
- 输入非法时，生成错误输出文件：
```json
{
  "success": false,
  "message": "input validation failed: ...",
  "source_file": "xxx.json"
}
```

## 5. 业务流程（文件驱动）

1. 外部系统向 `runtime/inbox` 写入输入文件。
2. `Worker` 轮询发现新文件并进行格式校验。
3. 校验通过后执行 `Evaluator.evaluate()`，生成标准输出写入 `runtime/outbox`。
4. `Supervisor` 轮询读取 `outbox` 中标准迭代输出。
5. `Supervisor` 调用 `BOEngine.update()` 更新历史数据。
6. 若未超过最大迭代次数，调用 `BOEngine.suggest_next()` 生成下一轮输入并写回 `runtime/inbox`。
7. 输入/输出原文件移动至 `runtime/processed` 归档。

## 6. 里程碑计划与验收标准

### Milestone 1（已完成）：MVP 框架搭建
- 验收标准：
  - 输入目录新文件可被自动检测。
  - 非法输入可被识别并生成错误输出。
  - 合法输入可产出标准输出。
  - 监督器可基于输出触发下一轮输入。

### Milestone 2（计划中）：Base BO 替换随机策略
- 工作内容：
  - 将 `engine.suggest_next` 从随机策略替换为 `SingleTaskGP + EI`。
  - 引入初始设计（LHS）和标准化处理。
- 验收标准：
  - 在 Branin 或 Rosenbrock 上较随机搜索更快收敛。

### Milestone 3（计划中）：失败处理与约束优化
- 工作内容：
  - 将仿真失败/发散转换为约束违例标签。
  - 接入 cEI 做可行域内优化。
- 验收标准：
  - 失败区域采样频率在迭代中明显下降。

### Milestone 4（按需）：多保真与隐空间扩展
- 工作内容：
  - MultiTaskGP + 成本感知采集函数。
  - VAE 隐空间编码/解码接口。
- 验收标准：
  - 在同预算下优于单保真或原始高维直接优化。

## 7. 测试计划

### 7.1 单元测试
- `validator`：字段缺失、类型错误、数值越界。
- `engine`：状态保存/加载、一致性。
- `file_io`：JSON读写与文件发现。

### 7.2 集成测试
- 场景 A：合法输入 -> 成功输出 -> 自动生成下一轮输入。
- 场景 B：非法输入 -> 错误输出 -> 不触发有效迭代更新。
- 场景 C：达到 `max_iterations` 后停止生成新输入。

### 7.3 基准测试（建议）
- 使用 Branin、Rosenbrock 作为 evaluator，验证优化收敛行为。

## 8. 风险与应对

- 风险：文件轮询在高吞吐时可能重复处理或竞态。
  - 应对：引入文件锁/原子重命名/消息队列替代。
- 风险：外部仿真执行失败导致数据污染。
  - 应对：统一错误码、失败重试、隔离失败样本。
- 风险：引擎状态单文件存储在多任务并发时冲突。
  - 应对：按 `task_id` 分状态文件或迁移至 SQLite。

## 9. 运维与部署建议

- 单机部署：两个独立进程（`worker` 与 `supervisor`）。
- 日志建议：增加结构化日志（JSON）并写入 `runtime/logs`。
- 配置管理：后续增加 `yaml` 配置（目录、维度、上下界、停止条件）。

## 10. 附录：后续实施建议（本计划外）

1. 优先把 `engine.py` 中随机策略替换成 BoTorch Base Route，这是最关键价值点。
2. 尽快定义“外部仿真适配器协议”（输入模板渲染、执行命令、结果解析），避免后续每个仿真器重复开发。
3. 增加 `task_id` 级别目录隔离，支持多任务并行运行。
4. 在 `outbox` 增加签名字段（如 `schema_version`、`producer`）提高跨程序兼容性。
5. 在进入高级分支前，先完成 Branin/Rosenbrock 回归测试与CI，保证框架稳定再叠加复杂算法。
