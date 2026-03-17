# SU2 翼型 BO 优化效果分析与问题定义说明

## 1. 背景与现象

当前任务配置为：

- task_id: `111`
- problem: `su2_airfoil_2d`
- strategy: `base_single_task_gp_ei`
- surrogate/acq/optimizer: `single_task_gp + expected_improvement + botorch_optimize_acqf`
- 变量维度: 5 维
- bounds: `[-0.06, 0.06] x 5`
- 最大迭代: 50
- 初始随机点: 8
- CFD 配置: `mesh_level=low`, `iterations=120`, `solver=EULER`, `processors=4`

从 `project/su/state/111_engine_state.json` 可见：

- 历史点数: 51（iter 0~50）
- 首次目标值: `-14.3019`
- 最优目标值: `-18.3734`（在较早阶段出现）
- 末次目标值: `-9.4722`
- 改善步数仅约 3 次，后续大部分阶段在较差区间波动

这和你在 UI 上看到的“蓝线波动大、红线长时间不更新”一致。

## 2. 为什么优化效果一般（重点从 BO 平台侧分析）

### 2.1 初始探索和样本效率问题

- 当前流程是严格串行单点评估（每轮 1 个样本）。
- 前 `initial_random_trials=8` 轮用于初始化探索，现已改为 Sobol 低差异序列采样，只有第 9 轮后才主要依赖代理模型建议。
- 在 CFD 高成本场景中，50 次预算里 8 次纯随机占比不低。

影响：早期样本对后续建模有较大“路径依赖”，且单点更新导致收敛速度偏慢。

### 2.2 代理模型与真实目标噪声不匹配

- 代理是 `single_task_gp` + EI，倾向于“平滑、低噪声”目标假设。
- 真实目标来自低网格 + 有限迭代的 CFD，存在明显数值噪声与离散误差。
- 这种“噪声目标 + 光滑先验”的错配，会让 EI 建议点在局部抖动，难稳定推进最优前沿。

影响：模型认为在“优化”，但真实仿真反馈并不一致，表现为目标值上下跳动。

### 2.3 目标函数形态对波动敏感

- 当前目标定义为 `objective = -(Cl / |Cd|)`。
- 比值目标对分母非常敏感，`Cd` 的微小波动会放大到 objective。
- 同时对 `Cd` 取绝对值会引入非光滑行为（在接近 0 区域更明显）。

影响：BO 接收到的标量目标曲面“陡峭且带噪”，拟合难度提高。

### 2.4 参数化自由度与可行域形状限制

- 虽然是 5 维变量，但几何生成带有强约束：固定 5 个 x 位置、上下表面存在耦合缩放、且有 clamp 与最小间隙约束。
- 实际可探索形状空间比“数学上 5 维盒约束”更窄、更不规则。

影响：代理优化在名义空间搜索，真实可行映射却可能非线性折叠，导致“看起来在探索，实际有效探索不足”。

### 2.5 保真度设置偏低，优化目标和最终目标存在偏差

- 当前 `mesh_level=low`、`iterations=120` 适合快速筛选，不适合精细比较。
- 低保真噪声会误导代理模型，产生“低保真最优”但非“高保真最优”。

影响：曲线可能较快找到一个早期最优，然后长时间难以稳定超越。

## 3. 结论

这次效果一般，不是单一参数导致，而是 **“高噪声 CFD 目标 + 比值型目标 + 串行单点 BO + 受约束参数化”** 的组合效应。  
从数据看，平台确实在迭代，但“有效优化步”偏少，表现为较长平台期和较大振荡。

## 4. 建议的改进方向

### 4.1 平台侧（优先）

- 将 `initial_random_trials` 从 8 调整为 6~10 的可配置策略，并采用 Sobol 初始化替代纯随机。
- 在 BO 阶段引入重复点评估或局部重采样，用于估计噪声水平。
- 在目标侧加入平滑/稳健形式，如 `-Cl/(Cd+eps)` 并固定 `eps`，避免分母过敏感。
- 增加“失败/异常点重试或降权”机制，避免单次数值异常污染 GP。

### 4.2 仿真侧

- 采用多保真：`low` 用于全局筛选，`medium` 用于候选复核。
- 对进入 BO 的 CFD 结果增加质量门槛（残差、物理量范围、收敛状态）。
- 对最优候选进行二次复算，减少“偶然最优”。

## 5. 已实施改造（当前代码状态）

- BO 初始化已接入 Sobol 采样，替代均匀随机初始化。
- SU2 评估改为两阶段：
  - 先在 `low` 网格计算；
  - 当 `low` 结果优于历史 `best_low_objective` 时，自动触发 `medium` 复核，并将 `medium` 结果回传 BO。
- SU2 结果目录按项目名分层保存到 `SU2/data/<project_name>/` 下。

---

## 6. 翼型问题基本信息（按你的要求放在文档末尾）

### 6.1 5 个控制点如何排布

- 默认弦向位置：`x = [0.1, 0.3, 0.5, 0.7, 0.9]`
- 每个设计变量 `p_i` 控制该位置厚度/弯度增量。
- 上下表面不是镜像，而是基于非对称基线：
  - upper base: `[0.03, 0.065, 0.07, 0.045, 0.018]`
  - lower base: `[-0.015, -0.022, -0.02, -0.012, -0.006]`
- 形变计算核心：
  - `delta = shape_scale * p_i`
  - `upper = clamp(upper_base + delta, -0.01, 0.16)`
  - `lower = clamp(lower_base + 0.45 * delta, -0.12, 0.03)`
  - 额外保持 `upper-lower >= min_gap(0.012)`

### 6.2 怎么优化（流程）

- BO 输出一组 5 维参数到 `inbox`。
- worker 调用 `plat_bo.objective.su2_airfoil_program`。
- 程序将参数写入 `SU2/data/<project_name>/<task_id>_iter_xxxx_trial_input.json` 并启动 SU2 pipeline。
- 每轮固定先跑 `low`，满足复核条件时再跑 `medium`，最终返回复核后的结果。
- pipeline 执行：几何生成 -> 网格 -> case 配置 -> SU2 求解 -> 结果提取 -> 可视化清单。
- 返回输出：`objective / objective_vector / Cl / Cd` 给 BO。
- supervisor 更新状态并生成下一轮输入。

### 6.3 输入/输出定义

- 输入（每轮）：
  - `task_id`
  - `iteration`
  - `parameters`（长度 5）
  - `problem_config`（网格级别、迭代步、SU2 路径、边界模式等）
- 输出（每轮）：
  - `objective`（当前默认 `-Cl/|Cd|`）
  - `objective_vector`（默认 `[-Cl, Cd]`）
  - `lift_coefficient`
  - `drag_coefficient`
  - `success/message`
  - `workspace`

### 6.4 按步保存的数据在哪里

- BO 侧：
  - `project/<project_id>/inbox/*_iter_xxxx_input.json`
  - `project/<project_id>/outbox/*_iter_xxxx_output.json`
  - `project/<project_id>/processed/*`
  - `project/<project_id>/state/<task_id>_engine_state.json`
  - `project/<project_id>/state/iterations/<task_id>_iter_xxxx_step.json`
- SU2 侧：
  - `SU2/data/<project_name>/<task_id>_iter_xxxx_low/`
  - `SU2/data/<project_name>/<task_id>_iter_xxxx_medium/`（触发复核时生成）
  - `SU2/data/<project_name>/<task_id>_iter_xxxx_trial_input.json`
  - `SU2/data/<project_name>/review_state.json`
  - 子目录内包含 `geometry.json, mesh.*, case.cfg, history.csv, flow.vtu, surface_flow.vtu, result.json, su2_stdout.log` 等
