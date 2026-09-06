# W&B stage gates

Use metrics within the same environment and reward scale. Prefer normalized
errors and fixed/frozen evaluators when comparing environments or checkpoints.

| Stage | Primary metrics | Better direction |
|---|---|---|
| Planner | `planner/score_normalized_rmse`, `planner/score_cosine_similarity`, `planner/loss` | RMSE/loss down; cosine up toward 1 |
| Reward | `reward/eval/normalized_mae`, `reward/eval/correlation`, `reward/eval/std_ratio`, `reward/eval/pred_negative_fraction` | MAE/negative fraction down; correlation up; std ratio toward 1 |
| Kernel | `kernel/eval/density_auc`, `kernel/eval/corrupted_rejection`, `kernel/eval/id_acceptance`, `kernel/train/nll` | AUC/rejection up; NLL down; ID acceptance near the configured quantile |
| Initial critic | `critic/eval/normalized_mae`, `critic/eval/normalized_bias`, `critic/eval/correlation`, `critic/eval/std_ratio` | errors down; correlation up; std ratio toward 1 |
| Critic warmup | `critic_warmup/normalized_mae`, `critic_warmup/normalized_bias`, `critic_warmup/std_ratio`, `critic_warmup/state_coverage`, `critic_warmup/feasible_plan_fraction` | errors down; std ratio toward 1; coverage/feasible fraction up |
| Finetune | `finetune/frozen/compositional_reward`, `finetune/frozen/base_reward`, `finetune/components/constraint_centered`, `finetune/components/objective_consistency_error` | frozen rewards up; centered constraint at or below 0; consistency error near 0 |
| Rollout | `finetune/rollout/success_rate`, `finetune/rollout/normalized_score`, `finetune/rollout/episode_return` | up on the same fixed seeds |

Useful starting gates (not universal constants): normalized MAE below `0.2`,
normalized bias below `0.1`, correlation above `0.8`, prediction/target standard
deviation ratio in `[0.8, 1.2]`, and warmup state coverage above `0.8`.

`finetune/loss` is an adjoint-matching fit loss, not task performance. `alpha`
is a schedule and `lambda` is a dual variable; neither is a quality score by
itself. Use frozen compositional reward together with the constraint and rollout
metrics to select a finetuned checkpoint.
