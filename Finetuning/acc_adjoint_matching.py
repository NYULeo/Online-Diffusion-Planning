'''Adjoint-matching fine-tuner for trajectory diffusion models (JAX/Flax port).

Ports the torch `Acc_AdjointMatchingFineTuner` to JAX/Flax in the FQL style. The frozen pretrained
score net (`old_score_net`) and the trainable control net (`new_score_net`) are held as
`jax_utils.TrainState`s; the lean-adjoint backward solve uses `jax.jvp` (was
`torch.autograd.functional.jvp`) and the per-step Jacobian helpers use `jax.jacrev` / `jax.vjp`.

The `accelerate.Accelerator` argument is preserved in the public `__init__` for API compatibility but
is treated as single-device here: `accelerate.utils.broadcast` is a no-op and the various
`split_between_processes` / `gather_for_metrics` / `reduce` / `unwrap_model` / `autocast` accelerator
methods are passed through unchanged (they degrade to identity on a single process).
'''
from __future__ import annotations
import functools
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Pretrain.Planners.Backbone.Dit import DiT1d
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
import wandb
from Finetuning.utils import Lambda, RewardDataset, PlannerDataset, KernelDataset, cycle, EMA, RewardTracker, karras_beta_schedule, clip_actions, save_planner, get_planner, getName, AlphaScheduler, AlphaSchedulerConfig
from Pretrain.Planners.Backbone.utils import cosine_alpha_sigma, cosine_beta, compute_dot_alpha_beta, get_pretrained_planner
from Pretrain.Dataset import get_PlannerName
from typing import Optional, Union
from Finetuning.traj_reward import RewardConfig, TotalReward, TotalReward_Critic
from Pretrain.Planners.Backbone.UNet import TemporalUnet
from Pretrain.Dataset import get_env
import copy
import pickle

from flax_utils import TrainState, target_update


# SPEED: when ODP_VMAP=1, the AM step processes all trajectories as ONE batched (B,H,dim) pass via
# jax.vmap instead of a 256x sequential Python loop. vmap(f) is mathematically identical to looping f
# (same math, same order per element), so this is logic-preserving; the sequential path stays the default.
_USE_VMAP = os.environ.get('ODP_VMAP', '0') == '1'


def broadcast(tensor, from_process=0):
    '''Single-device no-op replacement for `accelerate.utils.broadcast`.

    On a single process there is nothing to broadcast, so the input is returned unchanged.
    '''
    return tensor


@dataclass
class Acc_AdjointMatchingConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    horizon: int
    d_s: Optional[int] = None
    d_a: Optional[int] = None
    dataset_name: Optional[str] = None
    specific_dataset: Optional[str] = None
    task_id: Optional[int] = None
    backbone_name: str = 'transformer'
    eta: float = 0.8
    diffusion_steps: int = 30
    num_karras: int = 2
    num_Loss_Clip_steps: int = 35
    s: float = 0.008  # cosine schedule offset used in base drift
    sigma_min: float = 0.01
    sigma_max: float = 30.0
    device = 'cpu'  # JAX places automatically; single-device, was torch.device('cuda'/'cpu')
    step_start_ema = 50
    ema_decay = 0.99
    update_ema_every = 6
    finetune_lr: float = 1e-4
    finetune_total_steps: int = 500
    per_round_steps: int = 100
    lam: float = 0.01
    eta_lam: float = 0.001
    batch_per_sample: int = 3
    reward_scaling_factor: float = 100000
    alpha_scheduler_config: Optional[AlphaSchedulerConfig] = None
    update_lambda_every = 3
    update_kernel: bool = False
    MaxEnt: bool = False
    Entropy_Scaling_Factor: float = 0.5

    save_freq = 10
    save_model_freq = 50
    log_freq = 10



class Acc_AdjointMatchingFineTuner:
    """
    Implements fine‑tuning via the adjoint matching algorithm for
    trajectory diffusion models with fixed initial state.

    Given a pretrained score network (frozen), a differentiable reward
    model and a trainable control network, this class simulates
    trajectories with a memoryless noise schedule, solves the lean
    adjoint backwards and computes the adjoint matching loss on the
    unclamped dimensions of the state.
    """

    def __init__(
        self,
        accelerator,
        planner_checkpoint: int,
        AMConfig: Acc_AdjointMatchingConfig,
        ) -> None:

        self.config = AMConfig
        self.accelerator = accelerator
        self.device = self.accelerator.device
        rank = self.accelerator.process_index
        # JAX has no global RNG; seed a per-process key (was torch.manual_seed(42 + rank)).
        self.rng = jax.random.PRNGKey(42 + rank)

        self.ema = EMA(self.config.ema_decay)
        self.t_asc = jnp.linspace(1.0, 0.0, self.config.diffusion_steps + 1)
        self.k = self.kt(self.t_asc)
        self.t_grid, self.beta_1, self.sigma_grid = karras_beta_schedule(self.config.diffusion_steps, self.config.sigma_min, self.config.sigma_max, self.device)
        self.beta_2 = cosine_beta(self.t_grid, s = self.config.s)
        # SPEED (logic-identical): the per-diffusion-step scalars dt[i] and beta_now[i] depend only on i
        # (identical for every trajectory), so materialize them ONCE as Python floats here. Otherwise
        # sample_Traj_karras calls .item() on them — a blocking device->host sync — 2x per step x
        # diffusion_steps x 256 trajectories (~5k stalls/AM-step) that serialize the async dispatch queue.
        self._dt = []
        self._beta_now = []
        for i in range(self.config.diffusion_steps):
            t_next = self.t_grid[i + 1] if i < self.config.diffusion_steps - 1 else 0.0
            self._dt.append(float(t_next - self.t_grid[i]))
            self._beta_now.append(float(self.beta_1[i] if i < self.config.num_karras else self.beta_2[i]))

        self.set_old_score_net(planner_checkpoint)
        self.set_new_score_net()
        # SPEED (logic-identical): the DiT forward is a pure fn (train=False, dropout=0, deterministic
        # Fourier emb), so jitting model_def.apply computes the SAME values in the SAME order — it just
        # compiles once per input shape and caches, instead of eagerly re-dispatching + re-autotuning the
        # matmuls on every call (the `dot_search_space` storm). old/new share the architecture, so one
        # jitted apply serves both; params are passed as a traced arg so grads still flow when needed.
        self._score_apply = jax.jit(self.old_score_net.model_def.apply)
        self.set_ema_model()
        self.set_optimizer_and_scheduler()
        self.set_alpha_scheduler()
        self.set_lambda()
        self.set_reward_tracker()
        self.Initial_Conds = []


    def _next_rng(self):
        '''Split the trainer's RNG and return a fresh subkey (state mutates in place).'''
        self.rng, subkey = jax.random.split(self.rng)
        return subkey

    def _score(self, score_net, x, t):
        '''Frozen DiT forward through the jitted apply. Identical to score_net(x, t) (same params, same
        math) but compiled/cached per shape instead of eagerly re-autotuned each call. Use ONLY where the
        call is frozen (no param-grad): sampling + the adjoint jvp. The gradient path keeps the plain call.'''
        return self._score_apply({'params': score_net.params}, x, t)

    def Accelerate_Prepare(self, dataloader, reward_model: Union[TotalReward, TotalReward_Critic], round: int):
         if round == 1:
              self.new_score_net, self.old_score_net, self.optimizer, self.scheduler, dataloader, reward_model = self.accelerator.prepare(self.new_score_net, self.old_score_net, self.optimizer, self.scheduler, dataloader, reward_model)
         else:
              dataloader, reward_model = self.accelerator.prepare(dataloader, reward_model)
         return dataloader, reward_model

    def set_ema_model(self):
        # EMA copy of the trainable params. The ema_model is a frozen (no-tx) TrainState holding a
        # deep copy of new_score_net's params; gradients never flow through it.
        ema_params = jax.tree_util.tree_map(lambda x: jnp.array(x), self.new_score_net.params)
        self.ema_model = TrainState.create(self.new_score_net.model_def, ema_params, tx=None)

    def set_lambda(self, beta: Optional[float] = None):
        if beta is None:
           self.Lam = Lambda(lam = self.config.lam, beta = 1.0, eta_lam = self.config.eta_lam)
        else:
           self.Lam = Lambda(lam = self.config.lam, beta = beta, eta_lam = self.config.eta_lam)

    def sync_lambda(self):
        lam_val = self.Lam.get_lam() if self.accelerator.is_main_process else 0.0
        lam_tensor = jnp.asarray(lam_val, dtype=jnp.float32)
        lam_tensor = broadcast(lam_tensor, from_process=0)
        self.Lam.set_lam(lam_tensor.item())

    def set_optimizer_and_scheduler(self, new_lr=None, new_alpha=None, new_steps=None):
         # Use provided values or fall back to config defaults
         lr = new_lr if new_lr is not None else self.config.finetune_lr
         steps = new_steps if new_steps is not None else self.config.finetune_total_steps

         # Optimizer: faithfully match torch `torch.optim.Adam(lr, weight_decay=1e-2)`. Torch's Adam uses
         # COUPLED L2 (adds wd*param to the gradient BEFORE the Adam moments) — NOT decoupled AdamW. So
         # compose add_decayed_weights BEFORE scale_by_adam (optax.adamw decouples it, which is a different
         # optimizer). Clip-by-norm(1.0) first mirrors accelerator.clip_grad_norm_ in step(); cosine LR.
         self.scheduler = optax.cosine_decay_schedule(lr, decay_steps=steps)
         tx = optax.chain(
             optax.clip_by_global_norm(1.0),         # accelerator.clip_grad_norm_(max_norm=1.0)
             optax.add_decayed_weights(1e-2),        # COUPLED L2 (torch Adam weight_decay), before adam
             optax.scale_by_adam(),                  # b1=0.9,b2=0.999,eps=1e-8 == torch Adam defaults
             optax.scale_by_learning_rate(self.scheduler),
         )
         # Rebuild the trainable TrainState with the new optimizer (was: new torch.optim.Adam over
         # new_score_net.parameters()). Params are preserved across optimizer resets.
         self.optimizer = tx
         self.new_score_net = TrainState.create(self.new_score_net.model_def, self.new_score_net.params, tx=tx)

    def set_alpha_scheduler(self):
        self.alpha_scheduler = AlphaScheduler(config=self.config.alpha_scheduler_config)

    def set_old_score_net(self, planner_checkpoint: int):
        state_dict = get_planner(self.config.dataset_name, self.config.specific_dataset, planner_checkpoint, self.config.task_id)
        #state_dict = get_pretrained_planner(self.config.dataset_name, self.config.specific_dataset, planner_checkpoint)
        if( self.config.dataset_name == 'kitchen'):
              model_def = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'pointmaze'):
              model_def = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'cube'):
              model_def = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        elif (self.config.dataset_name == 'ogpointmaze'):
              model_def = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        else:
              raise ValueError(f"Invalid Environment: {self.config.dataset_name}")
        # TODO(checkpoint-bridge): `state_dict` comes from a torch planner checkpoint via
        # get_planner(...).load_state_dict(state_dict). Map the torch state_dict to the flax param tree
        # (Dense weight->kernel transposed, LayerNorm weight->scale, etc.) before building this state.
        params = state_dict
        # Frozen pretrained net: held as a no-tx TrainState, always called without params= (no grad).
        self.old_score_net = TrainState.create(model_def, params, tx=None)

    def set_new_score_net(self):
         if(self.config.backbone_name == 'transformer'):
              model_def = DiT1d(
                   in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128,
                   d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
              # Initialize from the frozen old_score_net's params (was: load_state_dict(old.state_dict())).
              new_params = jax.tree_util.tree_map(lambda x: jnp.array(x), self.old_score_net.params)
              self.new_score_net = TrainState.create(model_def, new_params, tx=None)
         elif(self.config.backbone_name == 'unet'):
              model_def = TemporalUnet(self.config.horizon, self.config.d_s + self.config.d_a)
              new_params = jax.tree_util.tree_map(lambda x: jnp.array(x), self.old_score_net.params)
              self.new_score_net = TrainState.create(model_def, new_params, tx=None)

    def set_reward_tracker(self):
        self.reward_tracker = RewardTracker(save_dir=f"./Finetuning/Results/{self.config.dataset_name}/{self.config.specific_dataset}/logs/")

    def step_ema(self, step):
        base_new_score_net = self.accelerator.unwrap_model(self.new_score_net)
        if step < self.config.step_start_ema:
            # Copy new params straight into the EMA model (was: ema.load_state_dict(new.state_dict())).
            ema_params = jax.tree_util.tree_map(lambda x: jnp.array(x), base_new_score_net.params)
            self.ema_model = self.ema_model.replace(params=ema_params)
            return
        # ema = decay * ema + (1 - decay) * new  ==  target_update(new, ema, tau=1 - decay).
        new_ema_params = target_update(base_new_score_net.params, self.ema_model.params, 1.0 - self.ema.beta)
        self.ema_model = self.ema_model.replace(params=new_ema_params)

    def reset_old_score_net(self, old_planner_checkpoint: int):
         state_dict = get_planner(self.config.dataset_name, self.config.specific_dataset,
                             old_planner_checkpoint, self.config.task_id)
         base = self.accelerator.unwrap_model(self.old_score_net)
         # TODO(checkpoint-bridge): map torch planner state_dict -> flax param tree before assigning.
         self.old_score_net = base.replace(params=state_dict)

    def set_new_score_net2(self):
         base_old = self.accelerator.unwrap_model(self.old_score_net)
         base_new = self.accelerator.unwrap_model(self.new_score_net)
         new_params = jax.tree_util.tree_map(lambda x: jnp.array(x), base_old.params)
         self.new_score_net = base_new.replace(params=new_params)

    def save_initial_conds(self, step: int):
        filename = 'Initial_Conds_' + str(step) + '.pkl'
        save_dir =  f"./Finetuning/Results/{self.config.dataset_name}/{self.config.specific_dataset}/"
        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(self.Initial_Conds, f)
        print(f"Initial Conditions saved to {save_path}")

    def vector_field(self, x: jnp.ndarray, t: jnp.ndarray, score_model) -> jnp.ndarray:
        # Compute beta(t) from cosine schedule
        k = jax.lax.stop_gradient(self.kt(t))
        v = k * x + k * score_model(x, t[None])
        return v

    def sigma_t(self, k: jnp.ndarray) -> jnp.ndarray:
        if(float(k) < 0):
           return jnp.sqrt(-2 * k)
        else:
           raise ValueError(f'K should be negative, but got {k.item()}')

    def kt(self, t: jnp.ndarray) -> jnp.ndarray:
       t = jnp.clip(t, 0.0, 1.0 - 1e-3)
       a = (math.pi / 2.0) * ((t + self.config.s) / (1.0 + self.config.s))
       return (-0.5) * (math.pi / (1.0 + self.config.s)) * jnp.tan(a)

    def compute_jacobian_vectorized(self, T, t_index):
       H_dim = self.config.horizon * (self.config.d_s + self.config.d_a)
       def score_fn(x_flat):
           x_reshaped = x_flat.reshape(T.shape)  # Reshape to original tensor shape
           score = self.old_score_net(x_reshaped, self.t_asc[t_index][None])
           return score.flatten()  # Return flattened score

       T_flat = T.flatten()

       # Use jax.jacrev for efficient computation (was torch.autograd.functional.jacobian).
       try:
           jacobian = jax.jacrev(score_fn)(T_flat)
           return jacobian
       except Exception as e:
           print(f"Warning: Vectorized Jacobian failed, falling back to element-wise: {e}")
           # Fallback to original method if vectorized fails
           return self._compute_jacobian_elementwise(T, t_index)

    def _compute_jacobian_elementwise(self, T, t_index):
       H_dim = self.config.horizon * (self.config.d_s + self.config.d_a)

       def score_fn(x_flat):
           x_reshaped = x_flat.reshape(T.shape)
           score = self.old_score_net(x_reshaped, self.t_asc[t_index][None])
           return score.reshape(-1)

       T_flat = T.flatten()
       # vjp with one-hot rows reproduces the per-output-element gradients of the torch loop.
       _, vjp_fn = jax.vjp(score_fn, T_flat)
       Jov = jnp.zeros((H_dim, H_dim))
       for j in range(H_dim):
           grad_outputs = jnp.zeros((H_dim,))
           grad_outputs = grad_outputs.at[j].set(1.0)
           # Compute gradient of j-th output w.r.t input
           grad_j = vjp_fn(grad_outputs)[0]
           # Store j-th row of Jacobian
           Jov = Jov.at[j, :].set(grad_j.reshape(-1))  # [H*dim]
       return Jov

    def sample_Traj(self,
        s0: jnp.ndarray,
        reward_model: Union[TotalReward, TotalReward_Critic],
        *, rng=None,
        ) ->  jnp.ndarray:
        rng = self._next_rng() if rng is None else rng

        s0_t = s0
        if ( (s0_t.shape[0] != self.config.d_s)   ):
             raise ValueError(f"s0 should have shape ({self.config.d_s},), but got {s0_t.shape[0]}")
        dim = self.config.d_s + self.config.d_a

        # Initialize x_T ~ N(0, I) with shape (horizon, dim)
        rng, k = jax.random.split(rng)
        x = jax.random.normal(k, (self.config.horizon, dim), dtype=jnp.float32)[None]
        conditions = s0_t[None]
        mask = jnp.zeros((1, self.config.horizon, dim), dtype=jnp.float32)
        mask = mask.at[:, 0, :self.config.d_s].set(1)
        y = jnp.zeros((1, self.config.horizon, dim), dtype=jnp.float32)
        y = y.at[:, 0, :self.config.d_s].set(conditions)
        #x = apply_conditioning(x, conditions, d_s)
        x = mask * y + (1 - mask) * x

        X = []
        X.append(x)
        for i in range(len(self.t_asc) - 1):
            t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            dt = (t_next - t_now).item()
            score = self._score(self.new_score_net, x, t_now[None])
            #drift = self.k[i] * x

            if self.config.eta > 0:
               rng, kn = jax.random.split(rng)
               noise = jax.random.normal(kn, x.shape, dtype=x.dtype)
               noise_scale = self.config.eta * jnp.sqrt((-2*self.k[i]) * (-dt))
               x = x + ((self.k[i] * x) +  (2*self.k[i] * score)) * dt + (noise_scale * noise)
            else:
               x = x + ((self.k[i] * x) +  (2*self.k[i] * score)) * dt

            x = mask * y + (1 - mask) * x
            X.append(x)
        #x = apply_conditioning(x, conditions, d_s)
        reward = reward_model._predict_jit(jnp.squeeze(X[-1], 0), self.Lam.get_lam())
        return  jnp.stack(X), reward

    def sample_Traj_karras(self,
        s0: jnp.ndarray, reward_model: Union[TotalReward, TotalReward_Critic],
        *, rng=None,
        ) ->  jnp.ndarray:
        rng = self._next_rng() if rng is None else rng

        s0_t = s0
        dim = self.config.d_s + self.config.d_a

        # Initialize x_T
        rng, k = jax.random.split(rng)
        x = jax.random.normal(k, (1, self.config.horizon, dim), dtype=jnp.float32) * self.sigma_grid[0]
        mask = jnp.zeros((1, self.config.horizon, dim), dtype=jnp.float32)
        mask = mask.at[:, 0, :self.config.d_s].set(1.0)
        y = jnp.zeros((1, self.config.horizon, dim), dtype=jnp.float32)
        y = y.at[:, 0, :self.config.d_s].set(s0_t[None])
        x = mask * y + (1 - mask) * x

        X = []
        X.append(x)
        for i in range(self.config.diffusion_steps):
             t_now = self.t_grid[i]
             dt = self._dt[i]            # precomputed Python float (was (t_next - t_now).item())
             beta_now = self._beta_now[i]  # precomputed Python float (was self.beta_{1,2}[i].item())
             # Drift
             drift = -0.5 * beta_now * x
             # Score
             score = self._score(self.new_score_net, x, t_now[None])
            # Euler step
             if self.config.eta > 0:
                 rng, kn = jax.random.split(rng)
                 noise = jax.random.normal(kn, x.shape, dtype=x.dtype)
                 noise_scale = self.config.eta * math.sqrt(beta_now * (-dt))
                 x = x + ((drift - beta_now * score) * dt + noise_scale * noise)
             else:
                 x = x + (drift - beta_now * score) * dt
             x = mask * y + (1 - mask) * x
             x = clip_actions(x, self.config.d_s)
             X.append(x)

        reward = reward_model._predict_jit(jnp.squeeze(X[-1], 0), self.Lam.get_lam())
        return jnp.stack(X), reward

    def make_a(self, X, reward_model: Union[TotalReward, TotalReward_Critic], reward_std: float):
        # X is a list of frozen trajectory states (no grad through the trajectory).
        X = [jnp.asarray(x) for x in X]
        steps_T = len(X)
        X_reversed = X[::-1]
        a = []
        T = X_reversed[0]
        T_squeezed = jnp.squeeze(T, 0)
        reward, gradient = reward_model._call_jit(T_squeezed, self.Lam.get_lam())
        #grad_norm = jnp.linalg.norm(gradient).clip(min=1e-8)
        #gradient = gradient * (1.0 / grad_norm)
        #print(f"Reward Gradeint Norm: {gradient.norm().item()}")
        if(self.config.MaxEnt):
            score = self._score(self.old_score_net, T, jnp.asarray(0.0)[None])
            EntGrad = -1 * score
            EntGrad = jax.lax.stop_gradient(EntGrad)
        else:
            EntGrad = jnp.expand_dims(jax.lax.stop_gradient(jnp.zeros_like(gradient)), 0)


        t_asc_reversed = jnp.flip(self.t_asc, axis=0)
        k_reversed = jnp.flip(self.k, axis=0)

        if(reward_std == 0.0):
            reward_std = 1.0
        #current_lr = self.optimizer.param_groups[0]['lr']
        alpha = self.alpha_scheduler.get_alpha()
        #a0 =  (-1 * ((self.config.reward_scaling_factor/alpha)/reward_std) * gradient).detach().unsqueeze(0).to(self.device) + (self.config.Entropy_Scaling_Factor * (-1) * EntGrad)
        a0 = jnp.expand_dims(jax.lax.stop_gradient(-1 * ((self.config.reward_scaling_factor/alpha/reward_std)) * gradient), 0) + (self.config.Entropy_Scaling_Factor * (-1) * EntGrad)
        #print(f"gradient norm: {gradient.norm().item()}")
        #max_norm = 5.0
        #a0 =   a0 * jnp.clip(max_norm / jnp.linalg.norm(a0), a_max=1.0)
        #print(f"a0: {a0.norm().item()}")
        # (removed a debug `if norm(a0).item()==0: print` — its .item() host-sync/Python branch is not
        # vmap/jit-safe and it had no effect on the computation.)

        a.append(a0)

        #a.append(jnp.expand_dims(jnp.zeros_like(gradient), 0))
        for i in range(steps_T - 1):
            #t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            t_now, t_next = t_asc_reversed[i], t_asc_reversed[i+1]
            dt = (t_now - t_next)
            #dt = (t_next - t_now)
            T = X_reversed[i]
            current_a = a[i]

            # jvp of the frozen old_score_net w.r.t. its input T, tangent = current_a (the t-input has
            # zero tangent). Was: torch.autograd.functional.jvp(self.old_score_net, (T, t_now[None]),
            #                                                    (current_a, zeros_like(t_now[None]))).
            y, jvp_out = jax.jvp(lambda x: self._score(self.old_score_net, x, t_now[None]), (T,), (current_a,))
            Jov_a = jvp_out
            new_a = current_a  + dt * ( (k_reversed[i] * current_a) + (2 * k_reversed[i] * Jov_a) )
            new_a = jax.lax.stop_gradient(new_a)
            a.append(new_a)
        a.reverse()
        return a, reward

    def adjoint_matching_loss(
        self,
        traj_x: List[jnp.ndarray],
        adjoints: List[jnp.ndarray],
        new_score_net=None,
    ) -> jnp.ndarray:
        # API-CHANGE: optional `new_score_net=` lets the jitted loss_fn pass a params-bound apply fn so
        # gradients flow into the trainable net; defaults to self.new_score_net (frozen call) otherwise.
        if new_score_net is None:
            new_score_net = self.new_score_net
        Loss = jnp.asarray(0.0)
        for i in range(len(traj_x)):
            traj_x_i = jax.lax.stop_gradient(traj_x[i])
            adjoint_i = jax.lax.stop_gradient(jnp.expand_dims(adjoints[i], 0).flatten())
            v_new = self.vector_field(traj_x_i, jax.lax.stop_gradient(self.t_asc[i]), new_score_net).squeeze(0).flatten()
            v_old = jax.lax.stop_gradient(self.vector_field(traj_x_i, jax.lax.stop_gradient(self.t_asc[i]), self.old_score_net).squeeze(0).flatten())
            sigma = jax.lax.stop_gradient(self.sigma_t(self.k[i]))
            if(i <= self.config.num_Loss_Clip_steps):
                Loss = Loss + jnp.minimum((((v_new - v_old)*(2/sigma) + (sigma * adjoint_i)) ** 2).mean(), jnp.asarray((self.config.reward_scaling_factor**2)*1.6))
            else:
                Loss = Loss + (((v_new - v_old)*(2/sigma) + (sigma * adjoint_i)) ** 2).mean()
        Loss = Loss / len(traj_x)
        return Loss


    def _step_vmapped(self, s0_batch, reward_model, rng):
        '''Batched equivalent of `step` for eta==0 (deterministic sampling): processes all
        S = len(s0_batch) * batch_per_sample trajectories as ONE (S,H,dim) pass instead of a Python loop.
        Every primitive mirrors the sequential path; reward/constraint use jax.vmap over the SAME per-traj
        predict/get_c/__call__ (vmap(f) == looping f). With eta==0 there is no per-step sampling noise, so
        the result is mathematically identical to the sequential loop (verify: same loss/reward/constraint).'''
        base = self.accelerator.unwrap_model(reward_model)
        lam = self.Lam.get_lam()
        d_s, d_a = self.config.d_s, self.config.d_a
        dim = d_s + d_a
        H = self.config.horizon
        # S0 repeated batch_per_sample times (matches the `for s0: for j in range(bps)` ordering).
        S0 = jnp.concatenate([jnp.repeat(s0_batch, self.config.batch_per_sample, axis=0)], axis=0)  # (S, d_s)
        S = S0.shape[0]

        # ---- sampling, batched over S. Replicate the sequential per-trajectory RNG EXACTLY so the result
        # is bit-identical to the loop (eta==0 => the only randomness is the init noise): step() chains rng
        # and per trajectory splits sk, then sample_Traj_karras splits k=split(sk)[1] and draws the init
        # noise. normal(k,(H,dim)) == normal(k,(1,H,dim)).squeeze(0) (same element order). ----
        keys = []
        r = rng
        for _ in range(S):
            r, sk = jax.random.split(r)
            _, k = jax.random.split(sk)
            keys.append(k)
        keys = jnp.stack(keys)
        noise0 = jax.vmap(lambda kk: jax.random.normal(kk, (H, dim), dtype=jnp.float32))(keys)  # (S,H,dim)
        x = noise0 * self.sigma_grid[0]
        mask = jnp.zeros((1, H, dim), dtype=jnp.float32).at[:, 0, :d_s].set(1.0)
        y = jnp.zeros((S, H, dim), dtype=jnp.float32).at[:, 0, :d_s].set(S0)
        x = mask * y + (1 - mask) * x
        X = [x]
        for i in range(self.config.diffusion_steps):
            t_now = self.t_grid[i]
            beta_now = self._beta_now[i]
            dt = self._dt[i]
            drift = -0.5 * beta_now * x
            score = self._score(self.new_score_net, x, jnp.broadcast_to(t_now, (S,)))
            x = x + (drift - beta_now * score) * dt           # eta==0 branch (no noise)
            x = mask * y + (1 - mask) * x
            x = clip_actions(x, d_s)
            X.append(x)
        X = jnp.stack(X)   # (steps+1, S, H, dim)

        finals = X[-1]                                         # (S, H, dim)
        rewards = jax.vmap(lambda xf: base._predict_jit(xf, lam))(finals)        # (S,)
        Cs = jax.vmap(lambda xf: base._get_c_jit(xf))(finals)                    # (S,)
        reward_std = float(jnp.max(rewards) - jnp.min(rewards))
        total_avgC = float(jnp.mean(Cs))
        if reward_std == 0.0:
            reward_std = 1.0

        # ---- adjoints, batched over S. make_a's recursion is elementwise over the leading dim, so the
        # SAME math runs with x of shape (S,1,H,dim). We replicate it directly (vmapping make_a would
        # re-call the reward __call__ per row anyway; here we batch that too). ----
        alpha = self.alpha_scheduler.get_alpha()
        Xr = X[:, :, None, :, :]                                # (steps+1, S, 1, H, dim) to match make_a's (1,H,dim)
        # reward gradient per trajectory via vmap over the SAME __call__ used sequentially.
        T_final = finals                                        # (S, H, dim)
        rg = jax.vmap(lambda xf: base._call_jit(xf, lam)[1])(T_final)   # (S, H, dim) gradients
        a_cur = jax.lax.stop_gradient(-1 * (self.config.reward_scaling_factor / alpha / reward_std) * rg)  # (S,H,dim)
        a_cur = a_cur[:, None]                                   # (S,1,H,dim); EntGrad==0 for MaxEnt=False
        t_asc_reversed = jnp.flip(self.t_asc, axis=0)
        k_reversed = jnp.flip(self.k, axis=0)
        X_reversed = X[::-1]                                    # (steps+1, S, H, dim)
        adjoints = [a_cur]
        for i in range(len(X) - 1):
            t_now = t_asc_reversed[i]
            dt = (t_asc_reversed[i] - t_asc_reversed[i + 1])
            # jvp of frozen old_score_net w.r.t. batched input; t broadcast to (S,) so the DiT noise dim
            # matches the batch exactly (no (1,)-vs-(S,) broadcast ambiguity). Same math as make_a.
            _, jov = jax.jvp(lambda z: self._score(self.old_score_net, z, jnp.broadcast_to(t_now, (S,))),
                             (X_reversed[i],), (jnp.squeeze(a_cur, 1),))
            jov = jov[:, None]
            a_cur = jax.lax.stop_gradient(a_cur + dt * (k_reversed[i] * a_cur + 2 * k_reversed[i] * jov))
            adjoints.append(a_cur)
        adjoints = adjoints[::-1]                                # list of (S,1,H,dim), len steps+1

        # ---- loss over params, batched. Inlines adjoint_matching_loss / vector_field with explicit (S,)
        # time so the trainable forward matches the batch exactly. Identical math to the sequential loss. ----
        def loss_fn(params):
            bound = functools.partial(self.new_score_net, params=params)
            Loss = jnp.asarray(0.0)
            n = len(X)
            for i in range(n):
                xi = jax.lax.stop_gradient(X[i])                 # (S,H,dim)
                ai = jax.lax.stop_gradient(jnp.squeeze(adjoints[i], 1).reshape(S, -1))
                tt = jax.lax.stop_gradient(self.t_asc[i])
                k_t = jax.lax.stop_gradient(self.kt(tt))
                tb = jnp.broadcast_to(tt, (S,))
                v_new = (k_t * xi + k_t * bound(xi, tb)).reshape(S, -1)
                v_old = jax.lax.stop_gradient(
                    (k_t * xi + k_t * self._score(self.old_score_net, xi, tb)).reshape(S, -1))
                sigma = jax.lax.stop_gradient(self.sigma_t(self.k[i]))
                per = (((v_new - v_old) * (2 / sigma) + (sigma * ai)) ** 2).mean(axis=1)   # (S,)
                if i <= self.config.num_Loss_Clip_steps:
                    per = jnp.minimum(per, jnp.asarray((self.config.reward_scaling_factor ** 2) * 1.6))
                Loss = Loss + per.mean()
            Loss = Loss / n
            return Loss, {'loss': Loss}

        grads, aux = jax.grad(loss_fn, has_aux=True)(self.new_score_net.params)
        self.new_score_net = self.new_score_net.apply_gradients(grads=grads)
        self.alpha_scheduler.step_alpha()
        avg_loss = float(aux['loss'])
        avg_reward = float(jnp.mean(rewards))
        return avg_loss, avg_reward, total_avgC


    def step(self, s0_batch: jnp.ndarray, reward_model: Union[TotalReward, TotalReward_Critic], *, rng=None) -> Tuple[float, float, float]:
        rng = self._next_rng() if rng is None else rng
        if _USE_VMAP and self.config.eta == 0:
            return self._step_vmapped(s0_batch, reward_model, rng)
        # 1. Split batch across processes
        base_reward_model = self.accelerator.unwrap_model(reward_model)
        with self.accelerator.split_between_processes(s0_batch) as local_s0:
            local_trajs = []
            local_final_Cs = []
            local_rewards = []
            for s0 in local_s0:
                s0 = s0
                #Mutiple Ones
                for i in range(self.config.batch_per_sample):
                   with self.accelerator.autocast():
                       rng, sk = jax.random.split(rng)
                       traj, reward = self.sample_Traj_karras(s0, base_reward_model, rng=sk)
                   #print(f"Reward: {reward.item()}")
                   #if(reward.item() == 0.0):
                       #continue

                   local_trajs.append(traj)
                   final_x = jnp.squeeze(traj[-1], 0)
                   C_val = base_reward_model._get_c_jit(final_x)
                   local_final_Cs.append(C_val)
                   local_rewards.append(reward)

            if(len(local_trajs) != 0):
                local_trajs = jnp.stack(local_trajs)
                local_final_Cs = jnp.stack(local_final_Cs)
                local_rewards = jnp.stack(local_rewards)
            else:
               # Create empty tensors with appropriate shape/device
               local_trajs = None
               local_final_Cs = jnp.asarray([0.0]*len(local_s0))
               local_rewards = jnp.asarray([0.0]*len(local_s0))
            #print(f"Local Trajs: {local_trajs.shape}")
            #print(f"local_trajs: {local_trajs}")
        self.accelerator.wait_for_everyone()

        local_Cs_det = jax.lax.stop_gradient(local_final_Cs)
        # 2. Gather C values and update lambda on main process
        all_final_Cs = self.accelerator.gather_for_metrics(local_Cs_det, use_gather_object = False)
        all_trajs = self.accelerator.gather_for_metrics(local_trajs, use_gather_object = False)
        all_rewards = self.accelerator.gather_for_metrics(local_rewards, use_gather_object = False)
        if self.accelerator.is_main_process:
            total_avgC = float(all_final_Cs.mean().item())
            #reward_std = float(all_rewards.std().item())
            reward_std = float(jnp.max(all_rewards).item() - jnp.min(all_rewards).item())

        else:
            total_avgC = 0.0
            reward_std = 0.0

        stats = jnp.asarray([total_avgC, reward_std])
        stats = broadcast(stats, from_process=0)
        total_avgC, reward_std = stats.tolist()
        self.accelerator.wait_for_everyone()


        # 3. Compute adjoints, rewards & loss tensors for each trajectory
        with self.accelerator.split_between_processes(all_trajs) as local_trajs:
        #if(local_trajs is not None):
            local_loss_tensors = []
            local_rewards = []

            # The adjoints are computed from the FROZEN old_score_net (no param-grad); the loss flows
            # gradients only through new_score_net's params. We therefore (a) precompute the adjoints
            # (stop-gradient), then (b) build a single loss_fn(params) over the whole local batch.
            traj_lists = []
            adjoint_lists = []
            for traj in local_trajs:
                traj = [traj[i] for i in range(traj.shape[0])]
                with self.accelerator.autocast():
                     adjoint, reward = self.make_a(traj, reward_model, reward_std)
                traj_lists.append(traj)
                adjoint_lists.append(adjoint)
                local_rewards.append(reward)

            def loss_fn(params):
                # Bind the trainable params so gradients flow through new_score_net (fql convention:
                # passing params= to the TrainState call flows gradients).
                bound_net = functools.partial(self.new_score_net, params=params)
                losses = []
                for traj, adjoint in zip(traj_lists, adjoint_lists):
                    losses.append(self.adjoint_matching_loss(traj, adjoint, new_score_net=bound_net))
                local_loss = jnp.stack(losses).mean()
                return local_loss, {'loss': local_loss}

            # For logging we also need the (detached) loss value per trajectory.
            for traj, adjoint in zip(traj_lists, adjoint_lists):
                loss_tensor = self.adjoint_matching_loss(traj, adjoint)  # detached (frozen apply)
                local_loss_tensors.append(loss_tensor)

            local_loss = jnp.stack(local_loss_tensors).mean()
            local_rewards = jnp.stack(local_rewards).mean()
        #else:
            #local_loss = jnp.asarray(0.0)
            #local_rewards = jnp.asarray(0.0)


        self.accelerator.wait_for_everyone()
        global_loss = self.accelerator.reduce(local_loss, reduction="mean")



        # 5. Backward and (maybe) optimizer step. In JAX gradients are functional: differentiate the
        # per-batch loss_fn w.r.t. new_score_net params and apply (grad clip is in the optax chain).
        with self.accelerator.accumulate(self.new_score_net):
            grads, _ = jax.grad(loss_fn, has_aux=True)(self.new_score_net.params)
            if self.accelerator.sync_gradients:
                self.new_score_net = self.new_score_net.apply_gradients(grads=grads)
                # scheduler is folded into the optax learning_rate (reads opt_state.count); no .step().
                self.alpha_scheduler.step_alpha()


         # 6. Logging: gather detached metrics
        local_loss_det = jax.lax.stop_gradient(local_loss)
        local_rewards_det = jax.lax.stop_gradient(local_rewards)
        all_losses = self.accelerator.gather_for_metrics(local_loss_det, use_gather_object=False)
        all_rewards = self.accelerator.gather_for_metrics(local_rewards_det, use_gather_object=False)

        #if self.accelerator.is_main_process:
        if self.accelerator.is_main_process:
             #if isinstance(all_losses, torch.Tensor):
            avg_loss = float(all_losses.mean().item())
            avg_reward = float(all_rewards.mean().item())
            return avg_loss, avg_reward, total_avgC
        return 0, 0, 0


    def finetune_planner(self, dataloader, reward_model: Union[TotalReward, TotalReward_Critic], round: int, old_planner_checkpoint: Optional[int] = None, *, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        if old_planner_checkpoint is not None:
            self.reset_old_score_net(old_planner_checkpoint)
            self.set_new_score_net2()
        # NOTE: was reward_model.eval() (torch). In JAX the reward/kernel/critic subnets are always
        # called frozen (apply without params=), so eval-mode is a no-op; the call is removed.

        if(round > 1):
            self.set_lambda(reward_model.get_beta())
            self.set_ema_model()
            # API-CHANGE: torch read the live decayed lr via self.optimizer.param_groups[0]['lr'];
            # optax has no param_groups, so the cosine schedule is rebuilt from config.finetune_lr over
            # the remaining steps. Numerically the lr restarts at the base value each round (verify).
            self.set_optimizer_and_scheduler(new_lr = self.config.finetune_lr, new_steps = self.config.finetune_total_steps - ((round-1)*self.config.per_round_steps))


        if self.accelerator.is_main_process:
             print(f"Starting Preparing")
        dataloader, reward_model = self.Accelerate_Prepare(dataloader, reward_model, round)
        self.accelerator.wait_for_everyone()
        dataloader = cycle(dataloader)
        if self.accelerator.is_main_process:
             print(f"Starting Finetuning")

        step = 0
        total_loss = 0.0
        total_reward = 0.0
        pure_reward = 0.0
        total_C = 0.0
        Lambda_C = 0.0
        #total_var_reward = 0.0


        #conds = next(dataloader)
        while step < self.config.per_round_steps:
             conds = next(dataloader)
             _t0 = time.time()
             loss, avg_reward, avg_C = self.step(conds, reward_model)
             if self.accelerator.is_main_process and os.environ.get('ODP_AM_TIMING', '0') == '1':
                 print(f"[AM] round {round} step {step}/{self.config.per_round_steps} "
                       f"took {time.time() - _t0:.1f}s (loss={float(loss):.4g})", flush=True)

             self.accelerator.wait_for_everyone()

             if self.accelerator.is_main_process:
                total_loss += loss
                total_reward += avg_reward
                total_C += avg_C
                Lambda_C += avg_C


                Reward = avg_reward + (self.Lam.get_lam() * avg_C)
                self.reward_tracker.log_reward(((round-1)*self.config.per_round_steps+step), Reward, avg_C)
                pure_reward += Reward


                if (step % self.config.update_lambda_every == 0) and (self.config.update_kernel):
                     self.Lam.update(Lambda_C / self.config.update_lambda_every)
                     Lambda_C = 0.0
                     print(f"step: {step}, lambda: {self.Lam.get_lam()}")

                if ((((round-1)*self.config.per_round_steps + step) % self.config.update_ema_every) == 0):
                     self.step_ema(((round-1)*self.config.per_round_steps + step))

                if ((step % self.config.log_freq) == 0):
                    print('---------------------------------------------------------')
                    if(step == 0):
                         print(f"round: {round}, step: {step}, loss {total_loss}")
                         print(f"round: {round}, step: {step}, total reward {total_reward}")
                         print(f"round: {round}, step: {step}, reward {pure_reward }")
                         print(f"round: {round}, step: {step}, constraint {total_C}")
                         print(f"round: {round}, step: {step}, alpha {self.alpha_scheduler.get_alpha()}")
                    else:
                         print(f"round: {round}, step: {step}, loss {total_loss / self.config.log_freq}")
                         print(f"round: {round}, step: {step}, total reward {total_reward / self.config.log_freq}")
                         print(f"round: {round}, step: {step}, reward {pure_reward / self.config.log_freq}")
                         print(f"round: {round}, step: {step}, constraint {total_C / self.config.log_freq}")
                         print(f"round: {round}, step: {step}, alpha {self.alpha_scheduler.get_alpha()}")
                    global_step = (round - 1) * self.config.per_round_steps + step
                    denom = 1.0 if step == 0 else float(self.config.log_freq)
                    if wandb.run is not None:
                        wandb.log({'finetune/loss': total_loss / denom,
                                   'finetune/total_reward': total_reward / denom,
                                   'finetune/reward': pure_reward / denom,
                                   'finetune/constraint': total_C / denom,
                                   'finetune/alpha': float(self.alpha_scheduler.get_alpha()),
                                   'finetune/lambda': float(self.Lam.get_lam()),
                                   'finetune/round': round}, step=global_step)
                    total_loss = 0.0
                    total_reward = 0.0
                    pure_reward = 0.0
                    total_C = 0.0


                if ((step % self.config.save_freq == 0) and (step!=0)):
                    model_name = getName(self.config.dataset_name, self.config.specific_dataset)
                    #model_name = get_PlannerName(self.config.dataset_name, self.config.specific_dataset)
                    self.reward_tracker.save_logs(f"{model_name}_step{((round-1)*self.config.per_round_steps+step)}_finetune_reward_logs.pkl")
                    self.reward_tracker.plot_reward_curve(
                    save_path=f"./Finetuning/Results/{self.config.dataset_name}/{self.config.specific_dataset}/logs/{model_name}_step{((round-1)*self.config.per_round_steps+step)}_finetune_reward_curve.png",
                    title=f"{model_name} of step {((round-1)*self.config.per_round_steps+step)} Finetuning Avg Reward",
                    show_constraint=True,
                    smooth_window=50,
                  )
             if(step % self.config.update_lambda_every == 0):
                 self.sync_lambda()

             step = step+1
             self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
             save_planner(self.ema_model, self.config.dataset_name, self.config.specific_dataset, (round*self.config.per_round_steps), task_id = self.config.task_id)
        self.accelerator.wait_for_everyone()
        self.accelerator.wait_for_everyone()
