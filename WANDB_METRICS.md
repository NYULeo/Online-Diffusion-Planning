# ODP W&B 指标说明与阶段放行标准

本文档用于在不等待最终完整 rollout 的情况下，判断 ODP pipeline 每个阶段是否训练正常，以及如何针对不同环境选择超参数。

基本原则：只在相同环境、数据处理、reward scale、horizon 和固定 seed 下直接比较原始 loss/reward。跨环境比较时，优先使用 normalized error、correlation、std ratio、coverage 和相对于初始 checkpoint 的提升。

## 一分钟速查表

| 阶段 | 首要指标 | 较好的方向 |
|---|---|---|
| Planner | `planner/score_normalized_rmse` | 越低越好；小于 1 表示优于零预测 |
| Planner | `planner/score_cosine_similarity` | 越接近 1 越好 |
| Reward | `reward/eval/normalized_mae` | 越低越好；可先以 `< 0.2` 为参考 |
| Reward | `reward/eval/correlation` | 越接近 1 越好；可先以 `> 0.8` 为参考 |
| Reward | `reward/eval/std_ratio` | 越接近 1 越好 |
| Kernel | `kernel/eval/density_auc` | 越接近 1 越好；0.5 约等于随机判断 |
| Kernel | `kernel/eval/corrupted_rejection` | 越高越好 |
| Initial critic | `critic/eval/normalized_mae` | 越低越好 |
| Initial critic | `critic/eval/normalized_bias` | 越低越好 |
| Initial critic | `critic/eval/std_ratio` | 越接近 1 越好 |
| Critic warmup | `critic_warmup/normalized_mae` | 越低越好 |
| Critic warmup | `critic_warmup/state_coverage` | 越高越好；可先要求 `> 0.8` |
| Critic warmup | `critic_warmup/target_clipped_fraction` | 越低越好 |
| Finetune | `finetune/frozen/compositional_reward` | 越高越好，最重要的离线 finetune 指标 |
| Finetune | `finetune/components/constraint_centered` | 应当 `<= 0` |
| Rollout | `finetune/rollout/success_rate` | 相同固定 seeds 下越高越好 |

## 通用回归指标

Reward 和 critic 会记录一组相同的回归诊断指标。

### `mae`

预测与 target 的平均绝对误差。

- 越低越好。
- 受 reward/value 的绝对尺度影响，不能直接跨环境比较。

### `normalized_mae`

定义为：

```text
MAE / target_std
```

- 越低越好。
- `< 0.1` 通常表示拟合较好，`0.1-0.2` 可作为初始可接受范围，`> 0.5` 通常需要检查。
- 这些范围只是起始经验值，不是所有环境的硬阈值。

### `bias` 与 `normalized_bias`

```text
bias = mean(prediction - target)
normalized_bias = abs(bias) / target_std
```

- 接近 0 最好。
- 正 bias 表示整体高估；负 bias 表示整体低估。
- `normalized_bias < 0.1` 可作为初始参考。

### `correlation`

Prediction 与 target 的 Pearson correlation。

- 越接近 1 越好。
- 接近 0 表示模型不能区分高低价值样本，即使 MAE 看起来不高，也可能只是学会预测平均值。
- 小于 0 表示价值排序方向错误。

### `std_ratio`

```text
prediction_std / target_std
```

- 接近 1 最好。
- 接近 0 表示预测坍缩成常数。
- 远大于 1 表示预测波动过强或训练不稳定。

### `pred_negative_fraction`

模型预测小于 0 的比例。

- 对当前 reward/value pipeline 通常越低越好。
- Reward 和 critic 下游存在非负 clipping；过多负预测会被截断，无法提供有效 guidance。

### `positive_mae` 与 `background_mae`

- `positive_mae`：target 大于 0 的样本误差。
- `background_mae`：target 不大于 0 的样本误差。

稀疏 reward 环境中，普通 transition 远多于成功附近 transition。如果 `background_mae` 很低但 `positive_mae` 很高，模型通常只学会了背景值，没有学会成功信号。

## Planner pretraining

Planner loss 是 diffusion score matching loss：对离线轨迹加噪后，模型学习正确的去噪 score。它衡量轨迹分布拟合能力，不直接等于任务成功率。

### `planner/loss`

- 总体下降并进入稳定平台较好。
- 它包含 diffusion 时间权重，不能与不同 schedule、维度或 horizon 的实验直接比较。

### `planner/score_normalized_rmse`

```text
score prediction RMSE / target score RMS
```

- 越低越好，是 planner 阶段最重要的指标。
- 约等于 1 表示接近始终预测 0。
- 小于 1 表示已经学到有效 score。
- 趋近 0 表示 score 数值拟合准确。

### `planner/score_cosine_similarity`

衡量预测 score 和 target score 的方向是否一致。

- 越接近 1 越好。
- 0 表示方向无关；小于 0 表示方向相反。
- Cosine 高但 normalized RMSE 高，表示方向基本正确但强度错误。

### `planner/score_pred_std` 与 `planner/score_target_std`

用于检查 prediction 是否坍缩或尺度错误。两者应处于相近数量级。

### `planner/gradient_norm`

这是 gradient clipping 之前的梯度范数，当前 clip threshold 为 1。

- 不是越低越好。
- 长期远大于 1 表示梯度经常被裁剪，可能需要降低 learning rate 或检查 loss scale。
- 很早就接近 0 且 loss 不改善，表示可能没有有效学习。

### Planner 放行条件

```text
score_normalized_rmse 持续下降并低于 1
score_cosine_similarity 持续上升
loss 进入平台且 gradient_norm 不爆炸
```

当前这些是训练分布诊断，不是独立 held-out validation。选择最终 planner checkpoint 时仍应结合后续 kernel 对生成轨迹的接受率和小规模固定-seed rollout。

## Reward model

Reward target 的构造流程是：

```text
环境 reward -> 乘 target_reward -> Gaussian/EMA smoothing -> 网络拟合每个 (state, action)
```

训练使用 Smooth L1 loss。

### `reward/loss`

- 越低通常越好，但稀疏 reward 下容易被大量背景 transition 主导。
- 不能只根据这个指标选择 sigma 或 checkpoint。

### `reward/eval/normalized_mae`

- 越低越好。
- 比原始 loss 更适合比较不同 reward scale。

### `reward/eval/correlation`

衡量 target 较高的状态是否也被模型预测为较高。

- 越接近 1 越好。
- 如果 MAE 较低但 correlation 接近 0，通常表示模型只学会预测平均值。

### `reward/eval/std_ratio`

- 接近 1 最好。
- 接近 0 表示 reward 输出坍缩，对 planner 无法提供有区分度的 gradient。

### `reward/eval/pred_negative_fraction`

- 越低越好。
- 下游会把负 reward 截成 0，负预测比例高意味着大量输出被浪费。

### `reward/eval/positive_mae`

成功信号及其 smoothing 区域的误差。对稀疏环境，它通常比整体 loss 更重要。

### Reward 放行条件

```text
eval/normalized_mae 下降
eval/correlation 上升且不接近 0
eval/std_ratio 接近 1
eval/positive_mae 可接受
eval/pred_negative_fraction 较低
```

调 sigma 时：sigma 太小会使有效 gradient 只存在于终点附近；sigma 太大会使整条轨迹 reward 过于相似。应联合选择 correlation、positive MAE 和 std ratio 较好的 sigma。

注意：当前 `reward/eval/*` 使用配置数据集检查拟合和校准，不是严格独立的 held-out 集，因此不能单独证明跨轨迹泛化。

## Transition kernel

Kernel 学习 `(state, action) -> next_state` 的条件概率分布。MoG kernel 的训练目标为：

```text
negative log likelihood + lambda_reg * disagreement regularization
```

### `kernel/train/nll`

- 在相同环境和数据上越低越好。
- 连续概率密度的 NLL 可能为负，不要用是否大于 0 判断好坏。

### `kernel/train/regularization`

MoG mode disagreement 相对于预测方差的加权 penalty。

- 很大表示 mixture modes 分歧过强。
- 接近 0 也不一定最好，可能表示 mode 坍缩。
- 必须结合 density AUC 和 corrupted rejection 判断。

### `kernel/train/member_loss_spread`

不同 ensemble member 的 loss 差异。

- 很大表示部分成员没有训练稳定。
- 较小表示成员训练状态一致。
- 过度接近 0 不能证明 uncertainty 合理，仍需检查 OOD discrimination。

### `kernel/eval/density_auc`

真实 transition 的 log density 高于打乱 next state transition 的概率。

- 越接近 1 越好，是 kernel 最重要的指标。
- 0.5 约等于随机判断。
- 小于 0.5 表示模型反而偏好 corrupted transition。

### `kernel/eval/corrupted_rejection`

打乱 next state 后，被当前数据分布 threshold 拒绝的比例。

- 越高越好。
- 很低表示 kernel 无法阻止明显错误的动力学 transition。

### `kernel/eval/density_separation`

```text
mean(ID log density) - mean(corrupted log density)
```

- 越大越好。

### `kernel/eval/id_acceptance`

真实 transition 通过 log-density threshold 的比例。

- 不是越高越好。
- 当前 threshold 来自评估分布 quantile；当 quantile 为 0.99 时，ID acceptance 应约为 0.99。
- 它主要是 calibration sanity check，不是单独的模型选择指标。

### `kernel/eval/mahalanobis_mean` 与 `mahalanobis_p99`

表示真实 transition 到预测分布的平均距离和尾部距离。

- 相同环境下通常越低越好。
- 会受状态维度影响，不能直接跨环境比较。

### Kernel 放行条件

```text
density_auc 上升并明显高于 0.5
corrupted_rejection 较高
density_separation 为正且稳定
train/nll 不再明显改善
id_acceptance 与目标 quantile 一致
```

## Initial critic

Initial critic 的 target 流程为：

```text
reward model prediction -> clip(reward, 0, inf) -> divide by Q_scale
-> GAE/bootstrap target -> symlog -> critic regression
```

### `critic/train/loss_symlog`

Symlog 空间的 Smooth L1 loss。

- 越低越好，但不是原始 value 单位的误差。
- Symlog 会压缩大 value，因此很小的 loss 不代表 decoded value 一定准确。

### `critic/train/decoded_normalized_mae`

对 critic 输出执行 `symexp` 后，与训练 target 比较。

- 越低越好。
- 训练阶段 decoded value 仍处于 `reward / Q_scale` 单位。

### `critic/eval/normalized_mae`

Evaluation 已乘回 `Q_scale`，再与环境 reward return 比较。

- 越低越好，是 initial critic 最重要的误差指标。

### `critic/eval/correlation`

衡量 critic 是否能正确排序高价值与低价值状态。

- 越接近 1 越好。
- 对 planner guidance，正确排序往往与绝对 value 精度同样重要。

### `critic/eval/std_ratio`

- 接近 1 最好。
- 接近 0 表示 critic 对所有状态输出相似 value。
- 远大于 1 表示 value 波动过强。

### `critic/train/running_target_mean` 与 `running_target_std`

训练 target 的指数滑动统计。它们是稳定性指标，不是直接的质量分数。

- 缓慢变化通常正常。
- 突然剧烈变化表示 reward scale、bootstrap critic 或数据分布不稳定。
- std 接近 0 表示 target 坍缩。

### Initial critic 放行条件

```text
eval/normalized_mae 下降
eval/normalized_bias 接近 0
eval/correlation 上升
eval/std_ratio 接近 1
running_target_std 不坍缩且不过度发散
```

Critic eval 是有限 horizon 环境 return proxy，而训练 target 还包含 reward model 和 bootstrap，因此两者不是完全相同的 target，但现在处于一致的 value scale。

## Critic warmup / planner7

Warmup 从 planner 生成候选轨迹，用 kernel 过滤不可行 plan，并针对多个 bootstrap horizon 构造：

```text
R_K = discounted reward + bootstrapped critic value
R_target = R_mean - rho * R_std
```

Target 随后会被截到非负并执行 symlog。

### `critic_warmup/feasible_plan_fraction`

```text
kernel-feasible plans / generated plans
```

- 在 kernel threshold 合理时越高越好。
- 如果 threshold 太宽松，这个指标接近 1 也没有意义。

### `critic_warmup/state_coverage`

```text
至少拥有一个 feasible plan 的不同起点 / 请求的起点数量
```

- 越高越好，比单纯 plan fraction 更重要。
- 可先以 `> 0.8` 作为放行参考。
- 很低表示 critic 只在少量容易状态上更新。

### `critic_warmup/unique_start_states`

真正进入当前 critic update 的不同起点数量，应尽量接近 warmup batch size。

### `critic_warmup/normalized_mae` 与 `normalized_bias`

当前 warmup prediction 和 target 都在 symlog 空间。

- 越低越好。
- 应结合 std ratio 判断，避免 critic 仅预测均值。

### `critic_warmup/std_ratio`

- 接近 1 最好。
- 很小表示 critic output 坍缩。

### `critic_warmup/reward_hat_mean` 与 `reward_hat_std`

Planner 轨迹上的 reward model 输出，已经完成非负 clipping 和 Q-scale 缩放。

- Mean 本身不是越高越好，主要看不同 checkpoint 是否稳定。
- Std 接近 0 表示 reward model 无法区分 planner 生成的轨迹。

### `critic_warmup/multi_horizon_return_mean`

报告中的平均多 horizon return，即 `R_mean`。它应保持有限并随训练稳定。

### `critic_warmup/multi_horizon_uncertainty`

报告中的 `R_std`，表示不同 bootstrap horizon 给出的 return 是否一致。

- 在平均 return 相近时越低越可靠。
- 很高可能表示 critic、reward model或 plan 内部动力学不一致。

### `critic_warmup/conservative_gap`

```text
rho * multi_horizon_uncertainty
```

这是从 `R_mean` 中扣除的保守项。如果它和 `R_mean` 同量级，rho 可能过强。

### `critic_warmup/target_clipped_fraction`

保守 target 小于 0，最终被截成 0 的比例。

- 越低通常越好。
- 很高可能表示 rho 太大、initial critic 太悲观、reward 太弱或 planner plan 质量太差。

### `sampling_seconds` 与 `plans_per_second`

只表示性能，不表示 critic 或 plan 质量。

### Warmup 放行条件

```text
state_coverage 上升并达到可接受范围
feasible_plan_fraction 在合理 kernel threshold 下稳定
normalized_mae 和 normalized_bias 下降
std_ratio 接近 1
multi_horizon_uncertainty 不发散
target_clipped_fraction 较低
```

## Finetune compositional reward

当前带 critic 的 finetune base reward 为：

```text
discounted immediate reward + discounted terminal critic value
```

再扣除 kernel constraint penalty。新日志将每一部分单独记录。

### `finetune/components/immediate_reward`

```text
sum_t gamma^t * predicted_reward_t
```

- 越高通常越好。
- 可能被 reward model exploit，不能脱离 constraint、frozen evaluator 和 rollout 单独使用。

### `finetune/components/terminal_value`

```text
gamma^(H-1) * Q_scale * symexp(critic(s_terminal))
```

- 越高通常表示 planner 到达了 critic 认为更有价值的末端状态。
- 如果它暴涨但 frozen reward 和 rollout 不升，通常表示 critic drift 或 exploitation。

### `finetune/components/base_reward`

```text
immediate_reward + terminal_value
```

即尚未扣 kernel penalty 的组合回报。

### `finetune/components/constraint_mean`

每个 transition 的平均 kernel violation：

```text
c_t = softplus(min_log_prob - log_density_t)
```

- Log density 高于 threshold 时，`c_t` 接近 0。
- Log density 低于 threshold 时，`c_t` 增大。
- 越低表示 plan 越符合 kernel 学到的数据流形。

### `finetune/components/constraint_centered`

```text
constraint_mean - delta
```

- `<= 0` 表示约束满足。
- `> 0` 表示约束违反。
- 当 beta 为 1 时，delta 约为 `0.693`。

如果该指标长期精确停在约 `-0.693`，意味着所有 kernel penalty 几乎为 0。它可能表示轨迹确实合理，也可能表示 `min_log_prob` 太宽松、constraint 没有发挥作用。

### `finetune/components/constraint_penalty_applied`

实际从 base reward 中扣除的 penalty。

- 大的正数表示严重扣分，应避免持续增大。
- 负数来自当前代码的 constraint slack bonus，不需要刻意追求无限降低。

### `finetune/components/compositional_reward`

```text
base_reward - actual constraint penalty
```

- 越高越好。
- 它与旧的 `finetune/reward` 应基本一致。

### `finetune/components/objective_consistency_error`

检查新分解出的 compositional reward 是否与原训练代码目标一致。

- 应接近 0。
- 它只验证日志正确性，不表示 planner 质量。

### `finetune/components/paper_compositional_reward`

按照 report 中 `mean(c_t)-delta` 的 constraint 定义计算。

当前训练代码在带 critic 时实际使用 `sum(c_t)-delta`，而 report 使用 mean。因此它可能与实际 compositional reward 不同。两者差距越大，表示 horizon normalization 的影响越强。

### Frozen evaluator

`finetune/frozen/*` 始终使用 finetune 开始时复制的 checkpoint-0 critic：

- `finetune/frozen/terminal_value`
- `finetune/frozen/base_reward`
- `finetune/frozen/compositional_reward`

在当前 offline 配置中 reward 和 kernel 固定，因此 frozen compositional reward 基本是固定 evaluator 下的比较指标。

- 越高越好。
- 它是 finetune 最重要的离线模型选择指标。
- 如果 live terminal value 上升而 frozen compositional reward 不升，通常是 critic 自身变化，不是 planner 真正改善。

### `finetune/loss`

这是 adjoint matching fit loss，不是负 reward。

- 同一 alpha、同一局部阶段内通常越低越好。
- 不能要求全程单调下降，因为 alpha 下降会放大 reward guidance，改变 target 的尺度。

### `finetune/alpha`

预设 schedule，不是质量指标。Alpha 越小，`reward_scaling_factor / alpha` 越大，guidance 越强。

### `finetune/lambda`

Constraint dual variable。

- 上升表示最近存在 constraint violation。
- 保持在初始值表示约束基本满足。
- 达到上限 5 是明显警告，表示 planner 持续离开 kernel 允许区域。
- Lambda 不是越高越好。

### Finetune 放行条件

```text
frozen/compositional_reward 上升
frozen/base_reward 上升或至少不下降
components/constraint_centered <= 0
components/constraint_penalty_applied 不持续恶化
rollout success/score/return 在固定 seeds 下不下降
```

## Rollout

### `finetune/rollout/success_rate`

成功 episode 的比例。

- 固定相同 seeds 和 episode 数量时越高越好。
- 样本少时方差很大。例如 32 个 episode 的最小变化单位是 `1/32 = 0.03125`，0.03 与 0.06 未必具有统计意义。

### `finetune/rollout/normalized_score`

环境 return 相对于 expert/reference score 的归一化结果。

- 同环境下越高越好。
- 不能直接跨环境比较。

### `finetune/rollout/episode_return`

每个 episode 的环境 reward 累积和。

- 同环境和 reward 定义下越高越好。
- 在成功率仍为 0 时，dense return 往往比 success rate 更早显示改善。

### `finetune/rollout/episode_length`

不能独立判断好坏：成功后快速结束时较短是好事，失败后提前终止时较短可能是坏事，一直失败直到 timeout 时通常很长。必须与 success rate 和 return 联合解释。

## 推荐的实际 W&B 检查顺序

```text
Planner:
  score_normalized_rmse down
  score_cosine_similarity up

Reward:
  eval/normalized_mae down
  eval/correlation up
  eval/std_ratio -> 1

Kernel:
  eval/density_auc up
  eval/corrupted_rejection up

Initial critic:
  eval/normalized_mae down
  eval/normalized_bias down
  eval/correlation up
  eval/std_ratio -> 1

Critic warmup:
  normalized_mae down
  normalized_bias down
  std_ratio -> 1
  state_coverage up
  target_clipped_fraction down

Finetune:
  frozen/compositional_reward up
  frozen/base_reward up
  constraint_centered <= 0
  rollout/success_rate or episode_return up
```

不要因为单个 train loss 更低就直接进入下一阶段。至少同时检查一个误差指标、一个分布/排序指标，以及一个与该阶段下游用途直接相关的指标。
