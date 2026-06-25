"""Sequential cube training pipeline for the JAX/Flax ODP port.

Runs the five training stages **in order**, each as its own Weights & Biases run, on a cube
variant (default `double`; choose another with `--variant`, e.g. `single`):

    1. pretrain  -> diffusion planner            (SDETrainer)                         env='cube', '<variant>-play'
    2. kernel    -> MoG transition kernel         (train_mog_kernel / test_kernel_mog) env='cube', '<variant>'
    3. reward    -> reward function               (train_reward / test_Model)          env='cube', '<variant>'
    4. critic    -> value/critic                  (train_critic / test_critic)         env='cube', '<variant>-play'
    5. finetune  -> adjoint-matching finetuning   (OnlineFinetuner.finetune_planner)   env='cube', '<variant>-play'

WHY two spellings of the variant: the kernel/reward/critic backbones take `--variant` verbatim
(e.g. `single`) while the planner, critic-rollout naming, and finetuner take `<variant>-<suffix>`
(e.g. `single-play` -> `Cube_SinglePlay`). Both resolve to the same checkpoint stem via `getName`, so
the stages chain correctly. This is the original ODP convention, preserved unchanged.

CHECKPOINT CHAINING: each stage saves at a step that the finetuner then loads. The step constants below
(PRETRAIN_STEPS, KERNEL_STEPS, REWARD_STEPS, CRITIC_STEPS) are passed both to the trainers (so they save
at that step) and to the FinetuningConfig (so it loads exactly those). Keep them consistent if you edit.

USAGE
    # cube SINGLE, full pipeline (online wandb; needs `wandb login` once):
    python run_cube_pipeline.py --variant single --task 4

    # cube double (default):
    python run_cube_pipeline.py

    # pick a subset of stages / resume mid-pipeline:
    python run_cube_pipeline.py --variant single --stages pretrain,kernel,reward,critic

    # quick smoke test with tiny step counts (verifies wiring end-to-end):
    python run_cube_pipeline.py --variant single --smoke

    # disable wandb (training is otherwise identical):
    python run_cube_pipeline.py --variant single --no-wandb
    # offline wandb (sync later with `wandb sync`):
    WANDB_MODE=offline python run_cube_pipeline.py --variant single

NOTE: requires the JAX stack + ogbench (see README "Installation"). Nothing here uses torch. Pretrained
torch checkpoints are NOT needed — every stage trains from scratch and saves a flax-serialized
checkpoint that the next stage / the finetuner reads (JAX-to-JAX). The legacy torch checkpoint-bridge
(docs/JAX_PORT_README.md) is only needed if you instead want to ingest the authors' original .pt weights.
"""
import argparse
import os
import sys
import time

# Make project-root imports work regardless of CWD (mirrors the entry scripts), and chdir to root so the
# relative ./Finetuning/... checkpoint paths used throughout the codebase resolve correctly.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import jax
import wandb

# --------------------------------------------------------------------------------------------------
# Environment + per-stage hyperparameters. These mirror the original cube-double entry-script values;
# only edit knowingly (the finetune stage loads checkpoints at the *_STEPS below).
# --------------------------------------------------------------------------------------------------
ENV_NAME = 'cube'
SPECIFIC_PLAY = 'double-play'   # planner / critic / finetune naming
SPECIFIC_DATA = 'double'        # kernel / reward / critic backbone dataset arg
TASK_ID = 4                     # cube-double task index (matches the example scripts)
HORIZON = 32
WANDB_PROJECT = 'odp-cube'

# Checkpoint steps each stage trains to / saves at (and that finetune then loads).
PRETRAIN_STEPS = 1_000_000
KERNEL_STEPS = 50_000
REWARD_STEPS = 100_000
CRITIC_STEPS = 50_000
FINETUNE_STEPS = 1_000_000

# Smoke-test overrides (tiny, just to exercise the full code path quickly).
SMOKE = dict(pretrain=200, kernel=200, reward=200, critic=200, finetune=200,
             kernel_save=200, reward_save=200, critic_save=10000, finetune_per_round=200)


def _banner(stage, run_name):
    print('\n' + '=' * 90)
    print(f'  STAGE: {stage}    (env={ENV_NAME}/{SPECIFIC_PLAY}, task={TASK_ID})    wandb run="{run_name}"')
    print('=' * 90, flush=True)


# Whether wandb runs are created this invocation (set in main() from --no-wandb).
_USE_WANDB = True


def init_run(name, group, config):
    """Start a wandb run for one stage (FQL-style `wandb.init`). No-op when --no-wandb is set."""
    if not _USE_WANDB:
        return
    wandb.init(project=WANDB_PROJECT, name=name, group=group, config=config, reinit=True)


def finish_run():
    """Finish the active wandb run (no-op if none / --no-wandb)."""
    if wandb.run is not None:
        wandb.finish()


# --------------------------------------------------------------------------------------------------
# Stage 1: pretrain the diffusion planner.
# --------------------------------------------------------------------------------------------------
def stage_pretrain(args, group):
    from Pretrain.utils import set_seed
    from Pretrain.Planners.Backbone.Trainer import SDETrainer

    num_steps = SMOKE['pretrain'] if args.smoke else PRETRAIN_STEPS
    init_run('pretrain', group,
                     config=dict(stage='pretrain', env=ENV_NAME, specific=SPECIFIC_PLAY, task_id=TASK_ID,
                                 horizon=HORIZON, num_steps=num_steps, batch_size=128, lr=2e-4))
    _banner('pretrain', 'pretrain')
    set_seed(args.seed)
    trainer = SDETrainer(
        ENV_NAME, SPECIFIC_PLAY, TASK_ID, HORIZON,
        backbone_name='transformer',
        num_steps=num_steps,
        batch_size=128,
        lr=2e-4,
        stride=1,
        seed=args.seed,
    )
    trainer.train()
    finish_run()


# --------------------------------------------------------------------------------------------------
# Stage 2: MoG transition kernel.
# --------------------------------------------------------------------------------------------------
def stage_kernel(args, group):
    from Pretrain.utils import set_seed
    from Pretrain.Transition_Kernel.Kernel_Backbone import train_mog_kernel, test_kernel_mog

    num_steps = SMOKE['kernel'] if args.smoke else KERNEL_STEPS
    save_freq = SMOKE['kernel_save'] if args.smoke else 10_000
    init_run('kernel', group,
                     config=dict(stage='kernel', env=ENV_NAME, specific=SPECIFIC_DATA, num_steps=num_steps,
                                 ensemble_size=10, num_modes=10, hidden_dim=514, num_hidden_layers=4))
    _banner('kernel', 'kernel')
    rng = set_seed(args.seed)
    rng, train_rng, test_rng = jax.random.split(rng, 3)
    train_mog_kernel(
        dataset_name=ENV_NAME,
        specific_dataset=SPECIFIC_DATA,
        batch_size=512,
        lr=1e-4,
        num_steps=num_steps,
        save_freq=save_freq,
        ensemble_size=10,
        num_modes=10,
        num_hidden_layers=4,
        hidden_dim=514,
        λ_reg=1e-3,
        noise_floor=5e-4,
        rng=train_rng,
    )
    test_kernel_mog(
        dataset_name=ENV_NAME,
        specific_dataset=SPECIFIC_DATA,
        trajs=None,
        save_freq=save_freq,
        num_steps=num_steps,
        num_hidden_layers=4,
        hidden_dim=514,
        ensemble_size=10,
        num_modes=10,
        quantile=0.99,
        noise_floor=5e-4,
        rng=test_rng,
    )
    finish_run()


# --------------------------------------------------------------------------------------------------
# Stage 3: reward function.
# --------------------------------------------------------------------------------------------------
def stage_reward(args, group):
    from Pretrain.utils import set_seed
    from Pretrain.Rewards.Reward_Backbone import train_reward, test_Model

    num_steps = SMOKE['reward'] if args.smoke else REWARD_STEPS
    save_freq = SMOKE['reward_save'] if args.smoke else REWARD_STEPS
    init_run('reward', group,
                     config=dict(stage='reward', env=ENV_NAME, specific=SPECIFIC_DATA, task_id=TASK_ID,
                                 num_steps=num_steps, hidden_dim=512, hidden_layers=4, sigma=4.0))
    _banner('reward', 'reward')
    rng = set_seed(args.seed)
    train_reward(
        dataset_name=ENV_NAME,
        hidden_layers=4,
        hidden_dim=512,
        batch_size=256,
        num_steps=num_steps,
        save_freq=save_freq,
        lr=1e-4,
        min_lr=5e-6,
        sigma=4.0,
        alpha=None,
        target_reward=300.0,
        specific_dataset=SPECIFIC_DATA,
        task_id=TASK_ID,
        traj_length=None,
        rng=rng,
    )
    test_Model(
        ENV_NAME,
        hidden_layers=4,
        hidden_dim=512,
        specific_dataset=SPECIFIC_DATA,
        trajs=None,
        sigma=4.0,
        alpha=None,
        target_reward=300.0,
        task_id=TASK_ID,
        traj_length=None,
        save_freq=save_freq,
        num_steps=num_steps,
    )
    finish_run()


# --------------------------------------------------------------------------------------------------
# Stage 4: critic / value function.
# --------------------------------------------------------------------------------------------------
def stage_critic(args, group):
    from Pretrain.utils import set_seed
    from Pretrain.Dataset import get_dataset
    from Pretrain.Critic.train_critic import train_critic, test_critic

    num_steps = SMOKE['critic'] if args.smoke else CRITIC_STEPS
    init_run('critic', group,
                     config=dict(stage='critic', env=ENV_NAME, specific=SPECIFIC_PLAY, task_id=TASK_ID,
                                 num_steps=num_steps, hidden_dim=512, hidden_layers=5, gamma=0.99,
                                 horizon=HORIZON, tau=0.005, sigma=8.0))
    _banner('critic', 'critic')
    rng = set_seed(args.seed)
    # Build the offline trajectories the critic regresses on (from the cube dataset).
    data = get_dataset(ENV_NAME, SPECIFIC_PLAY, task_id=TASK_ID, traj_length=200)
    trajs = data.get_trajectories()

    train_critic(
        dataset_name=ENV_NAME,
        specific_dataset=SPECIFIC_PLAY,
        hidden_layers=5,
        hidden_dim=512,
        batch_size=256,
        num_steps=num_steps,
        gamma=0.99,
        horizon=HORIZON,
        lr=1e-5,
        min_lr=1e-6,
        tau=0.005,
        goal=None,
        sigma=8.0,
        target_reward=50.0,
        trajs=trajs,
        task_id=TASK_ID,
        rng=rng,
    )
    test_critic(
        dataset_name=ENV_NAME,
        specific_dataset=SPECIFIC_PLAY,
        hidden_layers=5,
        hidden_dim=512,
        checkpoint_step=num_steps,
        gamma=0.99,
        horizon=HORIZON,
        goal=None,
        sigma=8.0,
        target_reward=50.0,
        trajs=trajs,
        task_id=TASK_ID,
    )
    finish_run()


# --------------------------------------------------------------------------------------------------
# Stage 5: adjoint-matching finetuning of the planner (loads the four checkpoints above).
# --------------------------------------------------------------------------------------------------
def stage_finetune(args, group):
    import random
    import numpy as np
    from Finetuning.Finetune_Backbone import (
        OnlineFinetuner, FinetuningConfig,
        Train_Reward_Config, Train_Kernel_Config, Train_Critic_Config,
    )
    from Finetuning.utils import AlphaSchedulerConfig
    from Finetuning.acc_adjoint_matching import Acc_AdjointMatchingConfig
    from Finetuning.traj_reward import RewardConfig

    num_steps = SMOKE['finetune'] if args.smoke else FINETUNE_STEPS
    init_run('finetune', group,
                     config=dict(stage='finetune', env=ENV_NAME, specific=SPECIFIC_PLAY, task_id=TASK_ID,
                                 finetune_steps=num_steps, finetune_lr=2e-4, finetune_batch_size=12,
                                 planner_ckpt=PRETRAIN_STEPS, reward_ckpt=REWARD_STEPS,
                                 kernel_ckpt=KERNEL_STEPS, critic_ckpt=CRITIC_STEPS))
    _banner('finetune', 'finetune')

    AMConfig = Acc_AdjointMatchingConfig(horizon=HORIZON)
    RWConfig = RewardConfig(beta=1.0, min_log_prob=150.0, explore=False)
    AlphaConfig = AlphaSchedulerConfig(alpha_start=1.0, alpha_end=1.0, total_steps=num_steps)

    FTConfig = FinetuningConfig(
        AMConfig=AMConfig,
        RewardConfig=RWConfig,
        AlphaConfig=AlphaConfig,
        dataset_name=ENV_NAME,
        specific_dataset=SPECIFIC_PLAY,
        planner_checkpoint=PRETRAIN_STEPS,
        reward_model_checkpoint=REWARD_STEPS,
        kernel_model_checkpoint=KERNEL_STEPS,
        critic_model_checkpoint=CRITIC_STEPS,
        train_reward_config=Train_Reward_Config(),
        train_kernel_config=Train_Kernel_Config(),
        train_critic_config=Train_Critic_Config(),
        finetune_steps=num_steps,
        finetune_batch_size=12,
        finetune_lr=2e-4,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    finetuner = OnlineFinetuner(FTConfig)
    finetuner.finetune_planner(seed=rng)
    finish_run()


STAGES = {
    'pretrain': stage_pretrain,
    'kernel': stage_kernel,
    'reward': stage_reward,
    'critic': stage_critic,
    'finetune': stage_finetune,
}
ORDER = ['pretrain', 'kernel', 'reward', 'critic', 'finetune']


def main():
    p = argparse.ArgumentParser(description='Sequential cube training pipeline (JAX/Flax ODP port).')
    p.add_argument('--stages', default=','.join(ORDER),
                   help='comma-separated subset of: ' + ','.join(ORDER) + ' (default: all, in order).')
    p.add_argument('--seed', type=int, default=1, help='random seed (threaded as a jax PRNGKey).')
    p.add_argument('--variant', default='double',
                   help="cube variant: single | double | triple | quadruple (default: double). "
                        "Kernel/reward use this name; planner/critic/finetune use '<variant>-<suffix>'.")
    p.add_argument('--suffix', default='play', choices=['play', 'noisy'],
                   help="cube dataset suffix for the planner/critic/finetune naming (default: play).")
    p.add_argument('--task', type=int, default=4, help='cube singletask task id (default: 4; valid 1-5).')
    p.add_argument('--smoke', action='store_true', help='tiny step counts to verify wiring end-to-end.')
    p.add_argument('--no-wandb', action='store_true', help='disable wandb (logging becomes a no-op).')
    p.add_argument('--wandb-group', default=None,
                   help='wandb group name tying the stage-runs together (default: cube-<variant>-<seed>).')
    args = p.parse_args()

    global _USE_WANDB, SPECIFIC_DATA, SPECIFIC_PLAY, TASK_ID
    _USE_WANDB = not args.no_wandb
    # cube uses two spellings: kernel/reward take the bare variant ('single'); planner/critic/finetune
    # take '<variant>-<suffix>' ('single-play'). Both resolve to the same checkpoint stem via getName().
    SPECIFIC_DATA = args.variant
    SPECIFIC_PLAY = f'{args.variant}-{args.suffix}'
    TASK_ID = args.task

    requested = [s.strip() for s in args.stages.split(',') if s.strip()]
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        p.error(f'unknown stage(s): {unknown}. valid: {ORDER}')
    # Always run in the canonical order regardless of how they were listed.
    selected = [s for s in ORDER if s in requested]

    group = args.wandb_group or f'cube-{args.variant}-seed{args.seed}'

    print(f'[pipeline] env={ENV_NAME}/{SPECIFIC_PLAY} task={TASK_ID} | stages={selected} | '
          f'seed={args.seed} | smoke={args.smoke} | wandb_group={group}')

    t0 = time.time()
    for stage in selected:
        s0 = time.time()
        try:
            STAGES[stage](args, group)
        except Exception:
            # Make sure a failed stage closes its wandb run before we surface the error.
            finish_run()
            print(f'[pipeline] stage "{stage}" FAILED after {time.time() - s0:.1f}s', flush=True)
            raise
        print(f'[pipeline] stage "{stage}" done in {time.time() - s0:.1f}s', flush=True)

    print(f'\n[pipeline] all stages {selected} complete in {time.time() - t0:.1f}s.')


if __name__ == '__main__':
    main()
