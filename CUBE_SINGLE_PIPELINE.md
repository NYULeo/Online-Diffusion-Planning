# ODP Cube-Single：从数据到 rollout 的直观说明

适用代码：`hkw@f32aab4`，基线 `Debugger`。前六节是首次阅读主线，最后两节留作 debug 索引。核心思路是：**planner 先模仿离线轨迹；reward 判断计划是否有用；kernel 判断计划是否像真实动力学；critic 补上 32 步以后的价值；finetune 再把 planner 推向“高价值且可行”的计划。**

```text
OGBench trajectory: observations[T,28], actions[T,5], rewards[T]
        ├─ 32 步窗口 ───────────────→ Planner：学会生成计划
        ├─ 单步 (s,a,r) ────────────→ Reward：学会给计划打分
        ├─ 单步 (s,a,s') ───────────→ Kernel：学会判断转移是否可信
        └─ 128 步 value 窗口 ───────→ Critic：学会估计状态的未来价值
                                           ↓
初始状态 → Planner 生成 [32,33] 计划 → Reward + Kernel + Critic 联合打分
                                           ↓
                              Adjoint finetune Planner → 环境 rollout
```

## 0. 原始数据到底是什么

一条轨迹是字典：`observations[T,28]`、`actions[T,5]`、`rewards[T]`。代码遇到 terminal 或原始 reward 为 0 时切断轨迹，并把下一时刻 reward 对齐给当前 `(s_t,a_t)`。四个模型各自重新计算 observation mean/std；**planner、reward、kernel、critic 的 normalization 不通用**，这是排查数值问题时最重要的事实。

## 1. Planner：先学“正常计划长什么样”

- **数据：**只用 task4 `single-play`。每条轨迹滑动切成 `[32,33]`：32 个时间点，每点是 28 维标准化状态加 5 维原始动作；另取首状态 `[28]` 作条件。
- **怎么训练：**给整条计划加不同程度的噪声，但始终把首状态固定；DiT 学习如何把噪声计划拉回离线数据中的计划。
- **优化目标：**生成结果应像专家/行为数据，而不是直接追求任务 reward。loss 衡量预测的去噪方向是否正确。
- **输出：**EMA planner checkpoint 0。它提供合理的初始能力，也为后续 finetune 提供起点。

直觉：planner 此时只会“模仿怎么移动”，并不知道哪条计划对 task4 最好。

## 2. Reward：学习“这一步对 task4 有多好”

- **数据：**task4 的 `single-play + single-noisy`。每个样本是 `(标准化 s[28], a[5]) → 标量 r`。
- **标签：**OGBench reward 先平移到非负，乘 500，再用 `sigma=4` 沿时间平滑；于是成功附近的若干步都有渐变信号，而非只在终点有信号。
- **怎么训练：**4×512 MLP 回归这个平滑 reward，使用 Smooth-L1。
- **输出：**给任意计划的每个 `(s_t,a_t)` 打分，finetune 和 critic 都使用它。

注意：标签非负不代表预测永远非负；网络最后一层是线性的，没有 ReLU。

## 3. Kernel：学习“这个状态转移是否可信”

- **数据：**非 task-specific 的全部 Cube Single `play + noisy`；每个样本是 `(标准化 s[28], a[5], 标准化 s'[28])`。
- **怎么训练：**10 个 ensemble 成员分别预测 10-component Gaussian mixture 的 next-state 分布。
- **优化目标：**真实 `s'` 在预测分布下概率尽量高，同时加很小的 mode disagreement penalty。
- **输出：**`log p(s'|s,a)`。若计划某一步 log-density 低于 `-110`，该转移会受到 soft penalty；critic warmup 则直接拒绝含这种转移的计划。

直觉：reward 可能偏爱“高分但物理上不可能”的状态，kernel 用离线动力学分布阻止这种利用。

## 4. Critic：估计计划终点之后还值多少

### 初训

- **数据：**task4 `single-play` 每条轨迹最后 200 步，再切成 `(states[128,28], predicted_rewards[128])`。
- **reward 来源：**不是环境原始 reward，而是 reward model 预测，随后 clip 到 `[-20,20]` 并除以 5。
- **怎么训练：**从一段 reward 加旧 target critic 的未来估计构造第一个状态的 value target；MLP 输入一个状态、输出一个标量，以 Smooth-L1 拟合。target critic 用 Polyak 慢更新。
- **归一化：**value target 被运行 mean/std 标准化，所以 critic 输出可正可负且通常接近零均值、单位方差。真正用于 finetune 的值是 `Q_std × V + Q_mean`。

### Planner warmup 与每轮更新

从离线状态出发，planner 每个状态尝试生成多条 `[32,33]` 计划；kernel 保留全程可行的计划；reward model 计算多种 horizon 的 return，末端由 critic bootstrap；同一初态的计划 target 取平均，再训练 critic。finetune 每轮结束后用新 planner 再做 20 个 critic steps，因此 planner 和 critic 交替改进。

## 5. Finetune：把“会模仿”变成“会选择高价值计划”

训练 DataLoader 不再提供完整专家计划，只提供 task4 离线轨迹尾部 100 步中的初始状态 `[28]`。一次 optimizer step 的实际过程是：

1. 当前 planner 为每个初态生成 8 条计划；每条最终计划为 `[32,33]`，并保存 10 次去噪过程。
2. Reward 对前 31 步打分；critic 给第 32 个状态补 terminal value；kernel 对 31 个转移加不可行惩罚。
3. 对联合分数求关于整条计划的梯度，得到“每个状态/动作应该往哪里移动才能提高计划价值”。
4. Adjoint/JVP 把这个终点梯度沿反向扩散过程传播。
5. 更新新 planner，使其去噪 vector field 朝这些高价值方向移动，同时以旧 planner 作为参考；保存 EMA。

配置为 30 rounds×3 steps，checkpoint 依次为 3、6、…、90。`offline=True` 表示每轮 rollout 只用于监控，产生的环境轨迹**不会**加入 reward/kernel/planner 训练；reward 与 kernel 固定，critic 每轮更新。

## 6. Rollout：planner 到底能不能控制真实环境

环境给当前真实状态，planner 生成 32 步计划；执行前若干动作后，再用新的真实状态重新规划。这是 receding-horizon control，而不是一次生成整集动作。

- finetune 内部：4 ranks×8 envs，chunk=31，最长 4000 步；打印环境 `success` 和折扣后的处理 reward。
- 最终 `Rollout2.py`：固定 checkpoint 81；100 次随机 planner 采样，但环境 seed 恒为 1；每次依次尝试 chunk 3/4/5/6，任一次成功即计成功。因此它测的是“同一环境初态、多种 chunk 重试后的成功率”，不是固定策略在 100 个环境 seed 上的成功率。

## Debug 时从现象反查

| 现象 | 优先检查 |
|---|---|
| planner loss 正常，但生成轨迹一开始就异常 | planner stats、task4 checkpoint、首状态条件、action 范围、sampler schedule |
| model reward 上升，但真实 success 下降 | reward extrapolation、kernel threshold、constraint/lambda、terminal critic 是否主导 |
| critic mean≈0、std≈1，或单个输出为负 | 通常是 target normalization 的预期；再检查配套 Q mean/std 是否来自同一 checkpoint |
| 第 1–2 round 与旧版本不同 | 初态顺序/RNG、BF16 batched forward/JVP、reward range、lambda 和 critic checkpoint |
| finetune 指标好，但最终 Rollout2 明显不同 | checkpoint 81 vs 最终 90、chunk 31 vs 3–6、环境 seed 和重试规则 |
| W&B 缺点或横轴错乱 | 当前 AM、critic、timing 共用且回退 `_step`；每轮 3 steps 又小于 `log_freq=10` |

## 当前已确认的语义风险

1. planner objective 对 31 个 constraint 求和，lambda 更新却用 constraint 均值，尺度不一致。
2. critic bootstrap 中原尺度 reward 与标准化 critic 输出直接相加，单位不一致。
3. 配置写 warmup 1000 steps、LR `1e-6→1e-9`；独立脚本实际为 100、`1e-4→1e-5`。
4. critic plan generation 每步重新创建 `RandomState(42)`，总是选择同一批初始状态。
5. W&B step 回退且没有记录每轮后两个 AM steps；当前曲线不足以完整验证训练。
