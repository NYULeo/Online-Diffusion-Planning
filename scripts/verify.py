"""Data-free verification harness for the JAX/Flax ODP port.

Runs WITHOUT any dataset (no ogbench/d4rl download): it feeds synthetic arrays to every network and
core function, so one run surfaces *all* remaining import / flax-dataclass / jax-API / shape errors at
once — instead of the training pipeline dying one stage at a time.

    python scripts/verify.py

Each check is isolated, so a failure is recorded and the harness keeps going. At the end it prints a
PASS/FAIL table; paste the FAIL section if anything breaks. This needs only the JAX stack (jax, flax,
optax, distrax, einops, numpy) — the same env you train in.
"""
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Default XLA GPU autotuning OFF (slow/unstable on some driver+jaxlib combos) — set before importing jax.
# Override with your own XLA_FLAGS, or ODP_AUTOTUNE=1 to keep autotuning on.
if 'XLA_FLAGS' not in os.environ and os.environ.get('ODP_AUTOTUNE', '0') != '1':
    os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

import jax
import jax.numpy as jnp

RNG = jax.random.PRNGKey(0)
RESULTS = []  # (name, ok, short_err)


def check(name, fn):
    """Run one check, record pass/fail, never abort the harness."""
    global RNG
    try:
        fn()
        RESULTS.append((name, True, ''))
        print(f'  PASS  {name}')
    except Exception as e:
        tb = traceback.format_exc()
        RESULTS.append((name, False, tb))
        print(f'  FAIL  {name}: {type(e).__name__}: {e}')


def init_apply(name, make_module, *example, methods=()):
    """init() a flax module on `example`, then apply() it, then a jax.grad pass; plus optional methods."""
    def _run():
        m = make_module()
        params = m.init(RNG, *example)
        out = m.apply(params, *example)
        # gradient smoke: scalar loss through the module's forward
        def loss(p):
            o = m.apply(p, *example)
            o = o[0] if isinstance(o, (tuple, list)) else o
            return jnp.sum(jnp.asarray(o) ** 2)
        jax.grad(loss)(params)
    check(name, _run)


print('=' * 80)
print('ODP JAX/Flax verification (synthetic inputs, no dataset)')
print('jax', jax.__version__, '| devices', jax.devices())
print('=' * 80)

# ----------------------------------------------------------------------------- 1. import every module
print('\n[1] import every core module')
MODULES = [
    'flax_utils',
    'Pretrain.utils', 'Pretrain.Dataset',
    'Pretrain.Planners.Backbone.BaseDiffusion', 'Pretrain.Planners.Backbone.utils',
    'Pretrain.Planners.Backbone.Dit', 'Pretrain.Planners.Backbone.UNet',
    'Pretrain.Planners.Backbone.Sampler', 'Pretrain.Planners.Backbone.Trainer',
    'Pretrain.Critic.nets', 'Pretrain.Critic.train_critic',
    'Pretrain.Rewards.nets', 'Pretrain.Rewards.Reward_Backbone',
    'Pretrain.Transition_Kernel.Kernel_Net', 'Pretrain.Transition_Kernel.Kernel_Backbone',
    'Finetuning.utils', 'Finetuning.traj_reward', 'Finetuning.adjoint_matching',
    'Finetuning.acc_adjoint_matching', 'Finetuning.Finetune_Backbone', 'Finetuning.Rollout',
]
import importlib
for mod in MODULES:
    check(f'import {mod}', lambda mod=mod: importlib.import_module(mod))

# ----------------------------------------------------------------------------- 2. networks: init/apply/grad
print('\n[2] network init / apply / grad (synthetic arrays)')
B, H, OBS, ACT = 2, 8, 5, 3
IN = OBS + ACT

from Pretrain.Planners.Backbone.Dit import DiT1d, DiT1Ref
for emb in ('positional', 'fourier'):
    init_apply(f'DiT1d[{emb}]',
               lambda emb=emb: DiT1d(in_dim=IN, emb_dim=16, d_model=32, n_heads=4, depth=2, timestep_emb_type=emb),
               jnp.zeros((B, H, IN)), jnp.zeros((B,)))
init_apply('DiT1Ref',
           lambda: DiT1Ref(in_dim=IN, emb_dim=16, d_model=32, n_heads=4, depth=2, timestep_emb_type='fourier'),
           jnp.zeros((B, H, 2 * IN)), jnp.zeros((B,)))

from Pretrain.Critic.nets import Critic, CriticEnsemble
init_apply('Critic', lambda: Critic(obs_dim=OBS, hidden_dim=32, hidden_layers=2), jnp.zeros((B, OBS)))
init_apply('CriticEnsemble', lambda: CriticEnsemble(obs_dim=OBS, hidden_dim=32, hidden_layers=2, num_heads=3),
           jnp.zeros((B, OBS)))

from Pretrain.Rewards.nets import SimpleReward, EnsembleReward
init_apply('SimpleReward', lambda: SimpleReward(OBS, ACT, 32, 2), jnp.zeros((B, OBS)), jnp.zeros((B, ACT)))
init_apply('EnsembleReward', lambda: EnsembleReward(OBS, ACT, 32, 2, ensemble_size=3),
           jnp.zeros((B, OBS)), jnp.zeros((B, ACT)))

from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel, MoGTransitionKernel
init_apply('RobustTransitionKernel', lambda: RobustTransitionKernel(OBS, ACT, 2, 32),
           jnp.zeros((B, OBS)), jnp.zeros((B, ACT)))
init_apply('MoGTransitionKernel', lambda: MoGTransitionKernel(OBS, ACT, num_modes=4, num_hidden_layers=2, hidden_dim=32),
           jnp.zeros((B, OBS)), jnp.zeros((B, ACT)))

from Pretrain.Planners.Backbone.UNet import TemporalUnet
def _unet():
    m = TemporalUnet(horizon=H, transition_dim=IN, dim=8, dim_mults=(1, 2))
    p = m.init(RNG, jnp.zeros((B, H, IN)), {}, jnp.zeros((B,)))
    m.apply(p, jnp.zeros((B, H, IN)), {}, jnp.zeros((B,)))
check('TemporalUnet', _unet)

# ----------------------------------------------------------------------------- 3. flax_utils plumbing
print('\n[3] flax_utils plumbing (TrainState / optax / target_update / ensemblize)')
def _trainstate():
    import optax
    from flax_utils import TrainState
    net = Critic(obs_dim=OBS, hidden_dim=16, hidden_layers=1)
    params = net.init(RNG, jnp.zeros((B, OBS)))['params']
    ts = TrainState.create(net, params, tx=optax.adam(1e-3))
    def loss_fn(p):
        out = ts(jnp.zeros((B, OBS)), params=p)
        return jnp.mean(out ** 2), {'loss': jnp.mean(out ** 2)}
    ts2, info = ts.apply_loss_fn(loss_fn)
    assert ts2.step == ts.step + 1
check('TrainState.create + apply_loss_fn', _trainstate)

def _target_update():
    from flax_utils import target_update
    a = {'w': jnp.ones((3,))}
    b = {'w': jnp.zeros((3,))}
    target_update(a, b, 0.5)
check('target_update', _target_update)

# ----------------------------------------------------------------------------- 4. kernel ensemble compute_*
print('\n[4] kernel ensemble compute_* (synthetic (model_def, params) tuples)')
def _kernel_compute():
    from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel, MoGTransitionKernel
    from Pretrain.Transition_Kernel.Kernel_Backbone import (
        compute_total_mahalanobis_score, compute_log_density,
        compute_log_density_mog, compute_total_mahalanobis_score_mog,
    )
    s, a, s_next = jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), jnp.zeros((B, OBS))
    robust = []
    for i in range(2):
        d = RobustTransitionKernel(OBS, ACT, 2, 32)
        robust.append((d, d.init(jax.random.PRNGKey(i), s, a)['params']))
    compute_total_mahalanobis_score(robust, s, a, s_next)
    compute_log_density(robust, s, a, s_next)
    mog = []
    for i in range(2):
        d = MoGTransitionKernel(OBS, ACT, num_modes=4, num_hidden_layers=2, hidden_dim=32)
        mog.append((d, d.init(jax.random.PRNGKey(10 + i), s, a)['params']))
    compute_log_density_mog(mog, s, a, s_next)
    compute_total_mahalanobis_score_mog(mog, s, a, s_next)
check('kernel compute_* (robust + mog ensembles)', _kernel_compute)

# ----------------------------------------------------------------------------- 5. samplers (synthetic score TrainState)
print('\n[5] diffusion samplers (synthetic planner TrainState)')
def _sampler():
    import optax
    from flax_utils import TrainState
    from Pretrain.Planners.Backbone.Dit import DiT1d
    from Pretrain.Planners.Backbone.Sampler import sample_euler_karras
    net = DiT1d(in_dim=IN, emb_dim=16, d_model=32, n_heads=4, depth=2, timestep_emb_type='fourier')
    params = net.init(RNG, jnp.zeros((1, H, IN)), jnp.zeros((1,)))['params']
    model = TrainState.create(net, params, tx=None)
    cur = jnp.zeros((OBS,))
    sample_euler_karras(cur, model, OBS, ACT, H, num_steps=4, num_karras=2, eta=0.8, device=None,
                        rng=jax.random.PRNGKey(1))
check('sample_euler_karras (4 steps)', _sampler)

# ----------------------------------------------------------------------------- summary
print('\n' + '=' * 80)
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
n_fail = len(RESULTS) - n_pass
print(f'SUMMARY: {n_pass}/{len(RESULTS)} passed, {n_fail} failed')
if n_fail:
    print('\n--- FAILURES (paste this whole block back) ---')
    for name, ok, tb in RESULTS:
        if not ok:
            print(f'\n### {name}\n{tb}')
    sys.exit(1)
print('ALL CHECKS PASSED ✅  — networks/plumbing are runtime-clean; safe to run scripts/smoke.sh next.')
