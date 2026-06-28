'''Trajectory reward + input-gradient model (reward - lambda * constraint), JAX/Flax port.

Computes the per-trajectory total reward and its gradient w.r.t. the trajectory tensor `x`
(state||action per step). The torch original used `torch.autograd.grad` with `grad_outputs` to
get the (vector-)Jacobian of the frozen reward / kernel / critic nets w.r.t. their *inputs*;
in JAX this is `jax.vjp` (CONVERSION_GUIDE §7). The nets themselves are frozen pretrained
checkpoints, so no parameter gradients flow — only input gradients.
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

from typing import Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from Pretrain.Rewards.nets import SimpleReward
from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel, MoGTransitionKernel
from Pretrain.Transition_Kernel.Kernel_Backbone import compute_log_density, compute_log_density_mog
from Pretrain.Critic.nets import Critic
from Finetuning.utils import get_reward_model, get_kernel, get_reward_stats, get_kernel_stats, get_critic_model, get_critic_stats

from flax_utils import TrainState


# ----------------------------------------------------------------------------------------------------------
# Infer a frozen net's architecture FROM its saved checkpoint, so we rebuild the exact module that was saved
# (regardless of config values or stale files). Avoids ScopeParamShapeError when loading reward/kernel ckpts.
# ----------------------------------------------------------------------------------------------------------
def _dense_kernels(state_dict):
    '''Every Dense weight ('kernel', 2-D) in a flax param state-dict, as a list of np arrays (LayerNorm has
    'scale'/'bias', not 'kernel', so it is naturally skipped).'''
    out = []

    def walk(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'kernel' and hasattr(v, 'shape') and np.asarray(v).ndim == 2:
                    out.append(np.asarray(v))
                else:
                    walk(v)

    walk(state_dict)
    return out


def _infer_reward_dims(state_dict, in_dim):
    '''SimpleReward(@nn.compact): Dense_0 (in_dim->hidden) .. hidden Denses .. final Dense(->1).
    hidden_dim = first Dense out-features; hidden_layers = (#Dense) - 2.'''
    ks = _dense_kernels(state_dict)
    first = next((k for k in ks if k.shape[0] == in_dim), ks[0])
    hidden_dim = int(first.shape[1])
    hidden_layers = max(len(ks) - 2, 0)
    return hidden_dim, hidden_layers


def _infer_mog_kernel_dims(state_dict, in_dim, obs_dim):
    '''MoGTransitionKernel: backbone Denses (all out=hidden_dim) + one head Dense
    (out=num_modes*(2*obs_dim+1)). Returns (hidden_dim, num_hidden_layers, num_modes). Order-independent:
    the head is the Dense whose out-dim != hidden_dim (falls back to the last Dense if they coincide).'''
    ks = _dense_kernels(state_dict)
    first = next((k for k in ks if k.shape[0] == in_dim), ks[0])
    hidden_dim = int(first.shape[1])
    head_candidates = [k for k in ks if int(k.shape[1]) != hidden_dim]
    head = head_candidates[-1] if head_candidates else ks[-1]
    num_modes = max(int(head.shape[1]) // (2 * obs_dim + 1), 1)
    num_hidden_layers = max(len(ks) - 1, 1)                  # all Denses except the head
    return hidden_dim, num_hidden_layers, num_modes


def _infer_robust_kernel_dims(state_dict, in_dim):
    '''RobustTransitionKernel: backbone Denses (->hidden_dim) + mean_head + log_std_head (->obs_dim each).'''
    ks = _dense_kernels(state_dict)
    first = next((k for k in ks if k.shape[0] == in_dim), ks[0])
    hidden_dim = int(first.shape[1])
    num_hidden_layers = max(len(ks) - 2, 1)                  # minus mean_head + log_std_head
    return hidden_dim, num_hidden_layers


def _infer_kernel_type(state_dict):
    '''Detect 'robust' vs 'mog' from the saved kernel's top-level submodule names, so the right kernel
    CLASS is rebuilt regardless of config (RobustTransitionKernel -> net_*/mean_head/log_std_head;
    MoGTransitionKernel -> backbone_*/head).'''
    keys = set(state_dict.keys()) if isinstance(state_dict, dict) else set()
    if 'mean_head' in keys or 'log_std_head' in keys or any(str(k).startswith('net_') for k in keys):
        return 'robust'
    return 'mog'


@dataclass
class RewardConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    beta: float
    min_log_prob: float
    quantile: float = 0.999
    number_of_generated_plans: int = 50
    explore: bool = True
    gamma: float = 0.8
    critic_gamma: float = 0.99
    device = None
    d_s: int = 0
    d_a: int = 0
    type_kernel: str = 'robust'
    kernel_num_modes: int = 8
    kernel_noise_floor: Optional[float] = 1e-4
    num_hidden_layers_kernel: int = 2
    hidden_dim_kernel: int = 256
    num_hidden_layers_reward: int = 1
    hidden_dim_reward: int = 128
    num_hidden_layers_critic: int = 1
    hidden_dim_critic: int = 128
    critic_d_s: int = 0
    delta: Optional[float] = None


def _torch_softplus(x, beta):
    '''Faithful port of torch.nn.functional.softplus(x, beta): (1/beta) * log(1 + exp(beta * x)).'''
    return jax.nn.softplus(beta * x) / beta


def _jax_norm_consts(stat):
    '''On-device (mean, 1/max(std, std_floor)) for an SAStats, so normalization is pure-JAX
    (s - mean) * inv_std == SAStats.norm_obs((s-mean)/max(std,std_floor)) but without a host round-trip.'''
    std = np.maximum(stat.obs_std, stat.std_floor)
    return (jnp.asarray(stat.obs_mean, dtype=jnp.float32), jnp.asarray(1.0 / std, dtype=jnp.float32))


# API-CHANGE: torch `nn.Module` base dropped. These classes were never trained as flax modules — they
# are orchestration objects holding frozen pretrained subnets and hand-rolling the input-gradient of the
# reward (CONVERSION_GUIDE §7). Public class names, constructor signatures and method signatures are
# identical; only the (unused) torch.nn.Module inheritance is gone.
class TotalReward:
    """
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int):
        super().__init__()
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
        self.config.device = device
        self.reward_net = SimpleReward(obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()
        self.kernels = []
        self.config.delta = F.softplus(torch.tensor(0.0, requires_grad = False), beta = self.config.beta).to(self.config.device)



        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        for i in range(len(kernel_state_dicts)):
                kernel_net = RobustTransitionKernel(obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel).to(self.config.device)
                kernel_net.load_state_dict(kernel_state_dicts[i])
                kernel_net.eval()
                self.kernels.append(kernel_net)


        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
        """

    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int, task_id: Optional[int] = None):
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint, task_id)
        self.config.device = device
        # TODO(checkpoint-bridge): `reward_state_dict` is a torch state_dict (Finetuning.utils.get_reward_model
        # still torch.load); map torch Linear weight (out,in)->flax kernel (in,out) transposed + LayerNorm
        # weight->scale before building params. We hold (model_def, params) and apply via TrainState.
        # Rebuild the reward net to match the SAVED checkpoint's dims (not config) so it always loads.
        r_hidden, r_layers = _infer_reward_dims(reward_state_dict, obs_dim + act_dim)
        reward_def = SimpleReward(obs_dim, act_dim, r_hidden, r_layers)
        reward_params = reward_state_dict  # torch state_dict; converted by the checkpoint bridge.
        self.reward_net = TrainState.create(reward_def, reward_params)
        self.kernels = []
        self.config.delta = _torch_softplus(jnp.asarray(0.0), self.config.beta)
        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        # Detect robust-vs-MoG from the saved checkpoint and set config.type_kernel so BOTH the rebuild
        # below and the consumer (sigmoid) use the kernel class that was actually trained.
        if kernel_state_dicts:
            self.config.type_kernel = _infer_kernel_type(kernel_state_dicts[0])
        if self.config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_hidden, k_layers = _infer_robust_kernel_dims(sd, obs_dim + act_dim)
                kernel_def = RobustTransitionKernel(
                    obs_dim, act_dim, k_layers, k_hidden,
                    noise_floor=self.config.kernel_noise_floor,
                )
                # TODO(checkpoint-bridge): `sd` is a torch kernel state_dict; remap to flax params.
                self.kernels.append(TrainState.create(kernel_def, sd))
        else:
            for sd in kernel_state_dicts:
                k_hidden, k_layers, k_modes = _infer_mog_kernel_dims(sd, obs_dim + act_dim, obs_dim)
                kernel_def = MoGTransitionKernel(
                    obs_dim, act_dim, k_modes, k_layers, k_hidden,
                    noise_floor=self.config.kernel_noise_floor,
                )
                # TODO(checkpoint-bridge): `sd` is a torch kernel state_dict; remap to flax params.
                # §11: the mog branch is consumed by compute_log_density_mog, which iterates
                # `for model_def, params in kernels`, so store (model_def, params) tuples here.
                self.kernels.append((kernel_def, sd))
        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)

        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)

        # SPEED (logic-identical): cache on-device (mean, 1/max(std,floor)) so the *_processor methods do
        # pure-JAX (s-mean)*inv_std instead of np.asarray()->norm_obs->jnp.asarray (a device<->host round
        # trip on EVERY per-step call: ~H x trajs x methods host syncs/AM-step). Same affine as SAStats.norm_obs.
        self._reward_norm = _jax_norm_consts(self.reward_stat)
        self._kernel_norm = _jax_norm_consts(self.kernel_stat)

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        if(not self.config.explore):
              self.config.gamma = 0.0

    def get_beta(self):
        return self.config.beta


    def sigmoid(self, s, a, s_next):
        if self.config.type_kernel == 'robust':
            total = jnp.array([0.0])
            for i in range(len(self.kernels)):
                mu, log_std = self.kernels[i](s, a)
                lp = self.kernels[i](s_next, mu, log_std, method='log_prob')
                total = total + lp
            avg = total / len(self.kernels)
        else:
            avg = compute_log_density_mog(self.kernels, s, a, s_next)
        x = self.config.min_log_prob - avg
        c = _torch_softplus(x, self.config.beta)
        return c


    def reward_processor(self, s):
        mean, inv_std = self._reward_norm   # pure-JAX (s-mean)/max(std,floor); no host round-trip
        return ((jnp.asarray(s, dtype=jnp.float32) - mean) * inv_std)

    def kernel_processor(self, s):
        mean, inv_std = self._kernel_norm
        return ((jnp.asarray(s, dtype=jnp.float32) - mean) * inv_std)

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[jnp.ndarray] = None):
        S = jnp.zeros((H, (self.config.d_s + self.config.d_a)))
        A = jnp.zeros((H, (self.config.d_s + self.config.d_a)))
        S = S.at[i, :self.config.d_s].set(s_grad)
        A = A.at[i, self.config.d_s:].set(a_grad)
        if s_next_grad is not None:
           S_next = jnp.zeros((H, (self.config.d_s + self.config.d_a)))
           S_next = S_next.at[i + 1, :self.config.d_s].set(s_next_grad)
           return S, A, S_next
        return S, A

    def get_c(self, x):
        H, D = x.shape
        C = jnp.asarray(0.0)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            a = x[i][self.config.d_s:][None]
            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            C += jnp.squeeze(c, 0)
        C = C / (H-1)
        C = C - self.config.delta
        return C

    def predict(self, x: jnp.ndarray, lam: float):
        H, D = x.shape
        total_reward = jnp.asarray(0.0)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s)[None]
            a = x[i][self.config.d_s:][None]


            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]


            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward += (1/H)*((self.config.critic_gamma**i)*(jnp.squeeze(r, 0))) - lam  * ( (1/(H-1)) * jnp.squeeze(c, 0))

        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        r = self.reward_net(s_norm_reward, a)
        total_reward +=  (1/H) * ((self.config.critic_gamma**(H-1))*(jnp.squeeze(r, 0)))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward

    def __call__(self, x: jnp.ndarray, lam: float):
        H, D = x.shape
        total_reward = jnp.asarray(0.0)
        gradient = jnp.zeros((H, D))
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s)[None]
            a = x[i][self.config.d_s:][None]


            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]


            # r = reward_net(s_norm_reward, a); grads of sum(r) w.r.t. (s_norm_reward, a) via vjp (§7).
            def reward_fn(sn, ac):
                return self.reward_net(sn, ac)
            r, reward_vjp = jax.vjp(reward_fn, s_norm_reward, a)
            grads = reward_vjp(jnp.ones_like(r))
            r_s = jnp.squeeze(grads[0], 0) * jnp.asarray(
                (1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
            r_a = jnp.squeeze(grads[1], 0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)



            # c = sigmoid(...); grads of sum(c) w.r.t. (s_norm_kernel, a, s_next_norm_kernel) via vjp (§7).
            def c_fn(sk, ac, snk):
                return self.sigmoid(sk, ac, snk)
            c, c_vjp = jax.vjp(c_fn, s_norm_kernel, a, s_next_norm_kernel)
            grads = c_vjp(jnp.ones_like(c))
            c_s = jnp.squeeze(grads[0], 0) * jnp.asarray(
                1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor), dtype=jnp.float32)
            c_a = jnp.squeeze(grads[1], 0)
            c_s_next = jnp.squeeze(grads[2], 0) * jnp.asarray(
                1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor), dtype=jnp.float32)
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient += (1/H)*((self.config.critic_gamma**i)*(r_s_grad + r_a_grad)) - lam * (1/(H-1)) * (c_s_grad + c_a_grad + c_s_next_grad)

            total_reward += (1/H)*((self.config.critic_gamma**i)*(jnp.squeeze(r, 0))) - lam  * ( (1/(H-1)) * jnp.squeeze(c, 0))
            #total_reward += (1/H)*(r.squeeze(0)) - lam * (1/(H-1)) * (c.squeeze(0) - self.config.delta)



        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]

        def reward_fn(sn, ac):
            return self.reward_net(sn, ac)
        r, reward_vjp = jax.vjp(reward_fn, s_norm_reward, a)
        grads = reward_vjp(jnp.ones_like(r))
        r_s = jnp.squeeze(grads[0], 0) * jnp.asarray(
            (1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
        r_a = jnp.squeeze(grads[1], 0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H-1)



        gradient += (1/H) * ((self.config.critic_gamma**(H-1))*(r_s_grad + r_a_grad))
        total_reward +=  (1/H) * ((self.config.critic_gamma**(H-1))*(jnp.squeeze(r, 0)))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient


class TotalReward_Critic:
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int, critic_checkpoint: int, task_id: Optional[int] = None):
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint, task_id)
        self.config.device = device
        # TODO(checkpoint-bridge): `reward_state_dict` is a torch state_dict; remap to flax params.
        # Rebuild reward net to match the SAVED checkpoint's dims (config-independent).
        r_hidden, r_layers = _infer_reward_dims(reward_state_dict, obs_dim + act_dim)
        reward_def = SimpleReward(obs_dim, act_dim, r_hidden, r_layers)
        self.reward_net = TrainState.create(reward_def, reward_state_dict)
        self.kernels = []
        self.config.delta = _torch_softplus(jnp.asarray(0.0), self.config.beta)


        critic_state_dict, critic_obs_dim = get_critic_model(dataset_name, specific_dataset, task_id, critic_checkpoint)
        # TODO(checkpoint-bridge): `critic_state_dict` is a torch state_dict; remap to flax params.
        # Critic shares SimpleReward's Dense layout (Dense_0 out=hidden, #Dense=hidden_layers+2), so infer
        # its dims from the saved checkpoint (config-independent) — input is just obs (no action concat).
        c_hidden, c_layers = _infer_reward_dims(critic_state_dict, critic_obs_dim)
        critic_def = Critic(critic_obs_dim, c_hidden, c_layers)
        self.critic = TrainState.create(critic_def, critic_state_dict)

        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        if kernel_state_dicts:
            self.config.type_kernel = _infer_kernel_type(kernel_state_dicts[0])
        if self.config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                k_hidden, k_layers = _infer_robust_kernel_dims(sd, obs_dim + act_dim)
                kernel_def = RobustTransitionKernel(
                    obs_dim, act_dim, k_layers, k_hidden,
                    noise_floor=self.config.kernel_noise_floor,
                )
                # TODO(checkpoint-bridge): `sd` is a torch kernel state_dict; remap to flax params.
                self.kernels.append(TrainState.create(kernel_def, sd))
        else:
            for sd in kernel_state_dicts:
                k_hidden, k_layers, k_modes = _infer_mog_kernel_dims(sd, obs_dim + act_dim, obs_dim)
                kernel_def = MoGTransitionKernel(
                    obs_dim, act_dim, k_modes, k_layers, k_hidden,
                    noise_floor=self.config.kernel_noise_floor,
                )
                # TODO(checkpoint-bridge): `sd` is a torch kernel state_dict; remap to flax params.
                # §11: the mog branch is consumed by compute_log_density_mog, which iterates
                # `for model_def, params in kernels`, so store (model_def, params) tuples here.
                self.kernels.append((kernel_def, sd))
        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)
        self.critic_stat = get_critic_stats(dataset_name, specific_dataset, task_id, 0)
        # SPEED (logic-identical): on-device norm constants -> pure-JAX processors (see TotalReward note).
        self._reward_norm = _jax_norm_consts(self.reward_stat)
        self._kernel_norm = _jax_norm_consts(self.kernel_stat)
        self._critic_norm = _jax_norm_consts(self.critic_stat)


        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        self.config.critic_d_s = critic_obs_dim
        if(not self.config.explore):
              self.config.gamma = 0.0

    def get_beta(self):
        return self.config.beta

    def sigmoid(self, s, a, s_next):
        if self.config.type_kernel == 'robust':
            total = jnp.array([0.0])
            for i in range(len(self.kernels)):
                mu, log_std = self.kernels[i](s, a)
                lp = self.kernels[i](s_next, mu, log_std, method='log_prob')
                total = total + lp
            avg = total / len(self.kernels)
        else:
            avg = compute_log_density_mog(self.kernels, s, a, s_next)
        x = self.config.min_log_prob - avg
        c = _torch_softplus(x, self.config.beta)
        return c


    def reward_processor(self, s):
        mean, inv_std = self._reward_norm   # pure-JAX (s-mean)/max(std,floor); no host round-trip
        return ((jnp.asarray(s, dtype=jnp.float32) - mean) * inv_std)

    def kernel_processor(self, s):
        mean, inv_std = self._kernel_norm
        return ((jnp.asarray(s, dtype=jnp.float32) - mean) * inv_std)

    def critic_processor(self, s):
        mean, inv_std = self._critic_norm
        return ((jnp.asarray(s, dtype=jnp.float32) - mean) * inv_std)

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[jnp.ndarray] = None):
        S = jnp.zeros((H, (self.config.d_s + self.config.d_a)))
        A = jnp.zeros((H, (self.config.d_s + self.config.d_a)))
        S = S.at[i, :self.config.d_s].set(s_grad)
        A = A.at[i, self.config.d_s:].set(a_grad)
        if s_next_grad is not None:
           S_next = jnp.zeros((H, (self.config.d_s + self.config.d_a)))
           S_next = S_next.at[i + 1, :self.config.d_s].set(s_next_grad)
           return S, A, S_next
        return S, A

    def makeGrad_Critic(self, H, s_grad, i):
        S = jnp.zeros((H, (self.config.d_s + self.config.d_a)))
        S = S.at[i, :self.config.critic_d_s].set(s_grad)
        return S

    def get_c(self, x):
        H, D = x.shape
        C = jnp.asarray(0.0)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            a = x[i][self.config.d_s:][None]
            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            C += jnp.squeeze(c, 0)
        C = C / (H-1)
        C = C - self.config.delta
        return C

    def predict(self, x: jnp.ndarray, lam: float):
        H, D = x.shape
        total_reward = jnp.asarray(0.0)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s)[None]
            a = x[i][self.config.d_s:][None]


            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]


            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward += ((self.config.critic_gamma**i)*(jnp.squeeze(r, 0))) - (lam  *  jnp.squeeze(c, 0))

        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        r = self.reward_net(s_norm_reward, a)
        final_s_critic = x[H-1][:self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic)[None]
        v = self.critic(final_s_norm_critic)
        #total_reward +=   ((self.config.critic_gamma**(H-1))*(r.squeeze(0))) + ( (self.config.critic_gamma**(H-1)) * v.squeeze(0))
        total_reward +=   ( (self.config.critic_gamma**(H-1)) * jnp.squeeze(v, 0))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward

    def __call__(self, x: jnp.ndarray, lam: float):
        H, D = x.shape
        total_reward = jnp.asarray(0.0)
        gradient = jnp.zeros((H, D))
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s)[None]
            a = x[i][self.config.d_s:][None]


            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]


            # r = reward_net(s_norm_reward, a); input-grad via vjp (§7).
            def reward_fn(sn, ac):
                return self.reward_net(sn, ac)
            r, reward_vjp = jax.vjp(reward_fn, s_norm_reward, a)
            grads = reward_vjp(jnp.ones_like(r))
            r_s = jnp.squeeze(grads[0], 0) * jnp.asarray(
                (1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
            r_a = jnp.squeeze(grads[1], 0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)



            # c = sigmoid(...); input-grad via vjp (§7).
            def c_fn(sk, ac, snk):
                return self.sigmoid(sk, ac, snk)
            c, c_vjp = jax.vjp(c_fn, s_norm_kernel, a, s_next_norm_kernel)
            grads = c_vjp(jnp.ones_like(c))
            c_s = jnp.squeeze(grads[0], 0) * jnp.asarray(
                1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor), dtype=jnp.float32)
            c_a = jnp.squeeze(grads[1], 0)
            c_s_next = jnp.squeeze(grads[2], 0) * jnp.asarray(
                1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor), dtype=jnp.float32)
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient += ((self.config.critic_gamma**i)*(r_s_grad + r_a_grad)) - (lam * (c_s_grad + c_a_grad + c_s_next_grad))

            total_reward += ((self.config.critic_gamma**i)*(jnp.squeeze(r, 0))) - (lam  * ( jnp.squeeze(c, 0)))
            #total_reward += (1/H)*(r.squeeze(0)) - lam * (1/(H-1)) * (c.squeeze(0) - self.config.delta)



        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]

        def reward_fn(sn, ac):
            return self.reward_net(sn, ac)
        r, reward_vjp = jax.vjp(reward_fn, s_norm_reward, a)
        grads = reward_vjp(jnp.ones_like(r))
        r_s = jnp.squeeze(grads[0], 0) * jnp.asarray(
            (1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
        r_a = jnp.squeeze(grads[1], 0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H-1)

        final_s_critic = x[H-1][:self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic)[None]

        def critic_fn(sc):
            return self.critic(sc)
        v, critic_vjp = jax.vjp(critic_fn, final_s_norm_critic)
        grads = critic_vjp(jnp.ones_like(v))
        v_s = jnp.squeeze(grads[0], 0) * jnp.asarray(
            (1/np.maximum(self.critic_stat.obs_std, self.critic_stat.std_floor)), dtype=jnp.float32)
        grad_critic = self.makeGrad_Critic(H, v_s, H-1)


        #gradient += ((r_s_grad + r_a_grad))  + ( (self.config.critic_gamma**(H-1)) * grad_critic)
        #total_reward +=  (r.squeeze(0)) + ((self.config.critic_gamma**(H-1)) * v.squeeze(0))
        gradient +=   ( (self.config.critic_gamma**(H-1)) * grad_critic)
        total_reward +=   ((self.config.critic_gamma**(H-1)) * jnp.squeeze(v, 0))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient
