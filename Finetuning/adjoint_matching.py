'''Adjoint matching fine-tuner for trajectory diffusion models (JAX/Flax port).'''
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple
import sys
import os

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Planners.Backbone.utils import (
    cosine_alpha_sigma, cosine_beta, compute_dot_alpha_beta, get_pretrained_planner,
)
from Finetuning.utils import Lambda, function
from traj_reward import RewardConfig, TotalReward
from Pretrain.Planners.Backbone.UNet import TemporalUnet
from Pretrain.Dataset import get_env

# Shared port plumbing (mirrors fql).
from JAX_PORT.jax_utils import (
    MLP, ModuleDict, TrainState, nonpytree_field, default_init, ensemblize,
    target_update, save_agent, restore_agent, supply_rng,
)




@dataclass
class AdjointMatchingConfig:
    """Configuration for the adjoint matching fine-tuner."""

    horizon: int
    lr: float = 2e-4
    d_s: int = 0
    d_a: int = 0
    backbone_name: str = 'transformer'
    eta: float = 0.8
    num_steps: int = 500
    s: float = 0.008  # cosine schedule offset used in base drift
    # torch held a `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` class
    # attribute here; JAX places arrays automatically, so the device field is dropped.
    lam: float = 1



class AdjointMatchingFineTuner:
    """
    Implements fine-tuning via the adjoint matching algorithm for
    trajectory diffusion models with fixed initial state.

    Given a pretrained score network (frozen), a differentiable reward
    model and a trainable control network, this class simulates
    trajectories with a memoryless noise schedule, solves the lean
    adjoint backwards and computes the adjoint matching loss on the
    unclamped dimensions of the state.
    """

    def __init__(
        self,
        env_name: str,
        specific_env: str,
        planner_checkpoint: int,
        reward_model_checkpoint: int,
        kernel_model_checkpoint: int,
        AMConfig: AdjointMatchingConfig,
        RewardConfig: RewardConfig
        ) -> None:
        self.config = AMConfig
        self.env, d_s, d_a = get_env(env_name, specific_env)
        self.config.d_s = d_s
        self.config.d_a = d_a
        # TODO(checkpoint-bridge): get_pretrained_planner returns the EMA planner params. In torch this
        # was a state_dict loaded via old_score_net.load_state_dict(state_dict); here it is the flax
        # param pytree ingested by the planner checkpoint bridge (CONVERSION_GUIDE §10).
        state_dict = get_pretrained_planner(env_name, specific_env, planner_checkpoint)
        if( env_name == 'kitchen'):
            old_score_def = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256,
                                  n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        elif (env_name == 'pointmaze'):
            old_score_def = DiT1d(in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128, d_model = 256,
                                  n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
        else:
          raise ValueError(f"Invalid Environment: {env_name}")
        # Frozen pretrained score net: no optimizer; params come from the checkpoint.
        self.old_score_net = TrainState.create(old_score_def, state_dict, tx=None)
        # TODO(checkpoint-bridge): TotalReward ingests torch reward/kernel checkpoints; ported elsewhere.
        self.reward_model = TotalReward(None, RewardConfig, env_name, specific_env,
                                        reward_model_checkpoint, kernel_model_checkpoint)
        self.backbone_selection()
        self.reset_parameters()
        self.t_asc = jnp.linspace(1.0, 0.0, self.config.num_steps + 1)
        self.k = self.kt(self.t_asc)
        self.Lam = Lambda(lam = self.config.lam, beta = self.reward_model.config.beta, eta_lam = self.config.lr)


    def vector_field(self, x: jnp.ndarray, t: jnp.ndarray, score_model, *, params=None) -> jnp.ndarray:
        # Compute beta(t) from cosine schedule
        k = self.kt(t)
        # `params` flows gradients through the (trainable) score model; None uses stored params (frozen).
        v = k * x + k * score_model(x, t[None], params=params)
        return v

    def reset_parameters(self):
        # torch: new_score_net.load_state_dict(old_score_net.state_dict()) -> copy frozen params in.
        self.new_score_net = self.new_score_net.replace(params=self.old_score_net.params)

    def get_C(self, x):
        # torch required grad of C w.r.t. x (x.requires_grad_(True)); compute it via jax.grad.
        x = x.squeeze(0)
        C = self.reward_model.get_c(x)
        return C

    def sigma_t(self, k: jnp.ndarray) -> jnp.ndarray:
        if(k < 0):
           return jnp.sqrt(-2 * k)
        else:
           raise ValueError(f'K should be negative, but got {k}')

    def kt(self, t: jnp.ndarray) -> jnp.ndarray:
       t = jnp.clip(t, 0.0, 1.0 - 1e-3)
       a = (math.pi / 2.0) * ((t + self.config.s) / (1.0 + self.config.s))
       return (-0.5) * (math.pi / (1.0 + self.config.s)) * jnp.tan(a)

    def compute_jacobian_vectorized(self, T, t_index):
       H_dim = self.config.horizon * (self.config.d_s + self.config.d_a)
       def score_fn(x_flat):
           x_reshaped = x_flat.reshape(T.shape)  # Reshape to original tensor shape
           score = self.old_score_net(x_reshaped, self.t_asc[t_index][None], condition=None)
           return score.flatten()  # Return flattened score

       T_flat = T.flatten()

       # jax.jacrev computes the input-Jacobian of the frozen score net (no param-grad).
       jacobian = jax.jacrev(score_fn)(T_flat)
       return jacobian

    def _compute_jacobian_elementwise(self, T, t_index):
       score = self.old_score_net(T, self.t_asc[t_index][None])
       H_dim = self.config.horizon * (self.config.d_s + self.config.d_a)
       Jov = jnp.zeros((H_dim, H_dim))

       def score_fn(x):
           return self.old_score_net(x, self.t_asc[t_index][None])

       _, vjp_fn = jax.vjp(score_fn, T)
       for j in range(H_dim):
           # Create one-hot for j-th output element
           grad_outputs = jnp.zeros_like(score)
           grad_outputs = grad_outputs.reshape(-1).at[j].set(1.0).reshape(grad_outputs.shape)

           # Compute gradient of j-th output w.r.t input (vector-Jacobian product)
           grad_j = vjp_fn(grad_outputs)[0]
           # Store j-th row of Jacobian
           Jov = Jov.at[j, :].set(grad_j.reshape(-1))  # [H*dim]
       return Jov

    def backbone_selection(self):
         if(self.config.backbone_name == 'transformer'):
              new_score_def = DiT1d(
                   in_dim = (self.config.d_s + self.config.d_a), emb_dim = 128,
                   d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
         elif(self.config.backbone_name == 'unet'):
              new_score_def = TemporalUnet(self.config.horizon, self.config.d_s + self.config.d_a)
         # Trainable score net: Adam(lr) with grad clipping (clip_grad_norm_(..., 1.0) applied in step()).
         tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(self.config.lr))
         self.new_score_net = TrainState.create(new_score_def, self.old_score_net.params, tx=tx)


    def sample_Traj(self,
        s0: jnp.ndarray,
        *, rng=None,
        ) ->  List[jnp.ndarray]:
        # API-CHANGE: added keyword-only `rng=` (torch used global RNG via torch.randn / randn_like).

        s0_t = s0
        if ( (s0_t.shape[0] != self.config.d_s)   ):
             raise ValueError(f"s0 should have shape ({self.config.d_s},), but got {s0_t.shape}")
        dim = self.config.d_s + self.config.d_a


        # Initialize x_T ~ N(0, I) with shape (horizon, dim)
        rng, k_init = jax.random.split(rng)
        x = jax.random.normal(k_init, (self.config.horizon, dim), dtype=jnp.float32)[None]
        conditions = s0_t[None]
        mask = jnp.zeros((1, self.config.horizon, dim), dtype = jnp.float32)
        mask = mask.at[:, 0, :self.config.d_s].set(1)
        y = jnp.zeros((1, self.config.horizon, dim), dtype = jnp.float32)
        y = y.at[:, 0, :self.config.d_s].set(conditions)
        #x = apply_conditioning(x, conditions, d_s)
        x = mask * y + (1 - mask) * x


        X = []
        X.append(x)
        for i in range(len(self.t_asc) - 1):
            t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            dt = (t_next - t_now).item()
            score = self.new_score_net(x, t_now[None])
            drift = self.k[i] * x

            if self.config.eta > 0:
               rng, k_noise = jax.random.split(rng)
               noise = jax.random.normal(k_noise, x.shape, dtype=x.dtype)
               noise_scale = self.config.eta * math.sqrt((-2*self.k[i]) * (-dt))
               x = x + (drift +  2*self.k[i] * score) * dt + noise_scale * noise
            else:
               x = x + (drift +  2*self.k[i] * score) * dt

            x = mask * y + (1 - mask) * x
            X.append(x)
        #x = apply_conditioning(x, conditions, d_s)
        return  X

    def make_a(self, X):
        steps_T = len(X)
        X_reversed = X[::-1]
        a = []
        T = X_reversed[0]
        T_squeezed = T.squeeze(0)
        reward, gradient = self.reward_model(T_squeezed, self.Lam.get_lam())
        gradient_flat = -1 * gradient.reshape(-1)  # [H*dim]
        a.append(gradient_flat)
        for i in range(steps_T - 1):
            t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            dt = (t_next - t_now)
            T = X_reversed[i]
            try:
                Jov = self.compute_jacobian_vectorized(T, i)
            except Exception as e:
                print(f"Vectorized Jacobian failed for step {i}, using fallback: {e}")
                Jov = self._compute_jacobian_elementwise(T, i)

            current_a = a[i]  # [H*dim]

            # Compute: a + dt * (k[i] * a + 2 * k[i] * Jov @ a)
            new_a = current_a + dt * (self.k[i] * current_a + 2 * self.k[i] * (Jov @ current_a))
            a.append(new_a)

        a.reverse()
        return a, reward.item()

    def adjoint_matching_loss(
        self,
        traj_x: List[jnp.ndarray],
        adjoints: List[jnp.ndarray],
        params=None,
    ) -> jnp.ndarray:
        # `params` flows gradients through new_score_net (the only trainable net in v_new).
        Loss = jnp.asarray(0.0)
        for i in range(len(traj_x)):
            traj_x_i = traj_x[i]
            adjoint_i = adjoints[i]
            v_new = self.vector_field(traj_x_i, self.t_asc[i], self.new_score_net,
                                      params=params).squeeze(0).flatten()
            v_old = self.vector_field(traj_x_i, self.t_asc[i], self.old_score_net).squeeze(0).flatten()
            sigma = self.sigma_t(self.k[i])
            Loss = Loss + jnp.sum(((v_new - v_old) * (2/sigma) + sigma * adjoint_i) ** 2)
        return Loss

    def step(self, s0: jnp.ndarray, *, rng=None) -> float:
        # API-CHANGE: added keyword-only `rng=` (sample_Traj draws noise; torch used global RNG).
        Total_C = 0.0
        Trajs = []
        for i in range(len(s0)):
            rng, k_traj = jax.random.split(rng)
            X = self.sample_Traj(s0[i], rng=k_traj)
            Trajs.append(X)
            x = X[len(X)-1].squeeze(0)
            c = self.get_C(x)
            Total_C += c
        avg_C = Total_C / len(s0)
        self.Lam.update(avg_C)

        total_reward = 0.0
        adjoints_all = []
        for i in range(len(s0)):
            adjoints, reward = self.make_a(Trajs[i])
            adjoints_all.append(adjoints)
            total_reward += reward
        avg_reward = total_reward / len(s0)

        # torch: zero_grad -> Loss.backward -> clip_grad_norm_(..., 1.0) -> optimizer.step().
        # jax: build the loss as a function of new_score_net params, take grads, apply (clip is in tx).
        def loss_fn(params):
            Loss = jnp.asarray(0.0)
            for i in range(len(s0)):
                Loss = Loss + self.adjoint_matching_loss(Trajs[i], adjoints_all[i], params=params)
            Loss = Loss / len(s0)
            return Loss

        Loss, grads = jax.value_and_grad(loss_fn)(self.new_score_net.params)
        self.new_score_net = self.new_score_net.apply_gradients(grads=grads)

        return Loss.item(), avg_reward, avg_C
