'''Trajectory-level reward + constraint model (JAX/Flax port of the PyTorch originals).

`TotalReward` / `TotalReward_Critic` wrap frozen, pre-trained reward / transition-kernel / critic
networks and expose:
  - `get_c(x)`               -> mean per-step constraint over the trajectory (minus delta),
  - `predict(x, lam)`        -> scalar reward-minus-lambda*constraint objective,
  - `__call__(x, lam)`       -> (total_reward, gradient) where `gradient` is the analytic input
                                gradient of `total_reward` w.r.t. the raw trajectory `x`.

§7 autograd-heavy: the per-step input gradients that torch computed with `torch.autograd.grad`
(outputs=r/c/v, inputs=normalized states/actions, grad_outputs=ones) are reproduced with
`jax.vjp` / `jax.grad`. The frozen pretrained nets are called via their `TrainState` without
`params=` (no param-gradients flow — §6). beta / min_log_prob and every other constant are kept
EXACT; only the differentiation backend changes.
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
import flax
import flax.linen as nn
import numpy as np
import optax

from Pretrain.Rewards.nets import SimpleReward
from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel
from Pretrain.Critic.nets import Critic
from Finetuning.utils import get_reward_model, get_kernel, get_reward_stats, get_kernel_stats, get_critic_model, get_critic_stats

# Shared port plumbing (mirrors fql).
from JAX_PORT.jax_utils import (
    MLP, ModuleDict, TrainState, nonpytree_field, default_init, ensemblize,
    target_update, save_agent, restore_agent, supply_rng,
)


def _softplus_beta(x, beta):
    '''torch F.softplus(x, beta) = (1/beta) * log(1 + exp(beta * x)) (numerically stable).'''
    return jax.nn.softplus(beta * x) / beta


@dataclass
class RewardConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    beta: float
    min_log_prob: float
    explore: bool = True
    gamma: float = 0.8
    critic_gamma: float = 0.99
    device = None
    d_s: int = 0
    d_a: int = 0
    num_hidden_layers_kernel: int = 2
    hidden_dim_kernel: int = 256
    num_hidden_layers_reward: int = 1
    hidden_dim_reward: int = 128
    num_hidden_layers_critic: int = 1
    hidden_dim_critic: int = 128
    critic_d_s: int = 0
    delta: Optional[float] = None


class TotalReward:
    # NOTE: torch defined this as `nn.Module`; in JAX it is a plain container holding frozen
    # pretrained networks as `(model_def, params)` TrainStates (it is never trained as a unit).
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int):
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
        self.config.device = device
        # TODO(checkpoint-bridge): torch did
        #   self.reward_net = SimpleReward(obs_dim, act_dim, hidden_dim_reward, num_hidden_layers_reward).to(device)
        #   self.reward_net.load_state_dict(reward_state_dict); self.reward_net.eval()
        # Map the torch state_dict into the flax SimpleReward param tree before building this TrainState.
        self.reward_net = self._make_state(
            SimpleReward(obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward),
            reward_state_dict,
        )
        self.kernels = []
        self.config.delta = _softplus_beta(jnp.asarray(0.0), self.config.beta)



        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        for i in range(len(kernel_state_dicts)):
                # TODO(checkpoint-bridge): torch loaded each kernel via load_state_dict(kernel_state_dicts[i]).
                kernel_net = self._make_state(
                    RobustTransitionKernel(
                        obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel),
                    kernel_state_dicts[i],
                )
                self.kernels.append(kernel_net)


        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)


        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        if(not self.config.explore):
              self.config.gamma = 0.0

    @staticmethod
    def _make_state(model_def, torch_state_dict):
        # TODO(checkpoint-bridge): build a TrainState whose params come from the torch state_dict.
        # A real remap (torch Linear weight (out,in) -> flax Dense kernel (in,out) transposed,
        # bias->bias, LayerNorm weight->scale) must populate `params`; here we keep the frozen-net
        # wrapper and leave the exact key-remap to the checkpoint bridge. `params=torch_state_dict`
        # is a placeholder so the public construction signature/behavior is preserved.
        return TrainState.create(model_def, params=torch_state_dict, tx=None)

    def get_beta(self):
        return self.config.beta

    def sigmoid(self, s, a, s_next):
        total = jnp.array([0.0])
        for i in range(len(self.kernels)):
            mu, log_std = self.kernels[i](s, a)
            lp = self.kernels[i](s_next, mu, log_std, method='log_prob')
            #lp = self.kernels[i].prob(s_next, mu, log_std)
            total = total + lp
        avg = total / len(self.kernels)
        x =  self.config.min_log_prob - avg
        c = _softplus_beta(x, self.config.beta)
        return c


    def reward_processor(self, s):
        s_n = np.asarray(s)
        s_n = self.reward_stat.norm_obs(s_n)
        s = jnp.asarray(s_n, dtype=jnp.float32)
        return s

    def kernel_processor(self, s):
        s_n = np.asarray(s)
        s_n = self.kernel_stat.norm_obs(s_n)
        s = jnp.asarray(s_n, dtype=jnp.float32)
        return s

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
            C += c.squeeze(0)
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
            total_reward += (1/H)*(r.squeeze(0)) - lam  * ( (1/(H-1)) * c.squeeze(0))

        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        r = self.reward_net(s_norm_reward, a)
        total_reward +=  (1/H) * (r.squeeze(0))
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


            # r and its input gradient w.r.t. (s_norm_reward, a): torch.autograd.grad(r, (s_norm_reward, a),
            # grad_outputs=ones) -> vjp of the reward net w.r.t. its inputs (§7).
            def reward_fn(sr, av):
                return self.reward_net(sr, av)
            r, reward_vjp = jax.vjp(reward_fn, s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)

            grads = reward_vjp(jnp.ones_like(r))
            r_s = grads[0].squeeze(0) * jnp.asarray(
                (1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
            r_a = grads[1].squeeze(0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)



            # c and its input gradient w.r.t. (s_norm_kernel, a, s_next_norm_kernel): create_graph=True in
            # torch (2nd-order capable); jax.vjp composes naturally if differentiated further (§7).
            def constraint_fn(sk, av, snk):
                return self.sigmoid(sk, av, snk)
            c, constraint_vjp = jax.vjp(constraint_fn, s_norm_kernel, a, s_next_norm_kernel)
            grads = constraint_vjp(jnp.ones_like(c))
            c_s = grads[0].squeeze(0) * jnp.asarray(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   dtype=jnp.float32)
            c_a = grads[1].squeeze(0)
            c_s_next = grads[2].squeeze(0) * jnp.asarray(
                1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor), dtype=jnp.float32)
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient +=  (1/H)*((r_s_grad + r_a_grad)) - lam * (1/(H-1)) * (c_s_grad + c_a_grad + c_s_next_grad)

            total_reward += (1/H)*(r.squeeze(0)) - lam  * ( (1/(H-1)) * c.squeeze(0))
            #total_reward += (1/H)*(r.squeeze(0)) - lam * (1/(H-1)) * (c.squeeze(0) - self.config.delta)



        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        def reward_fn(sr, av):
            return self.reward_net(sr, av)
        r, reward_vjp = jax.vjp(reward_fn, s_norm_reward, a)


        grads = reward_vjp(jnp.ones_like(r))
        r_s = grads[0].squeeze(0) * jnp.asarray((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)),
                                                dtype=jnp.float32)
        r_a = grads[1].squeeze(0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H-1)



        gradient += (1/H) * ((r_s_grad + r_a_grad))
        total_reward +=  (1/H) * (r.squeeze(0))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient


class TotalReward_Critic:
    # NOTE: torch defined this as `nn.Module`; in JAX it is a plain container holding frozen
    # pretrained networks as `(model_def, params)` TrainStates (it is never trained as a unit).
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int, critic_checkpoint: int):
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
        self.config.device = device
        # TODO(checkpoint-bridge): torch did SimpleReward(...).load_state_dict(reward_state_dict).eval()
        self.reward_net = self._make_state(
            SimpleReward(obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward),
            reward_state_dict,
        )
        self.kernels = []
        self.config.delta = _softplus_beta(jnp.asarray(0.0), self.config.beta)


        critic_state_dict, critic_obs_dim = get_critic_model(dataset_name, specific_dataset, critic_checkpoint)
        # TODO(checkpoint-bridge): torch did Critic(...).load_state_dict(critic_state_dict).eval()
        self.critic = self._make_state(
            Critic(critic_obs_dim, self.config.hidden_dim_critic, self.config.num_hidden_layers_critic),
            critic_state_dict,
        )

        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        for i in range(len(kernel_state_dicts)):
                # TODO(checkpoint-bridge): torch loaded each kernel via load_state_dict(kernel_state_dicts[i]).
                kernel_net = self._make_state(
                    RobustTransitionKernel(obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel),
                    kernel_state_dicts[i],
                )
                self.kernels.append(kernel_net)

        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)
        self.critic_stat = get_critic_stats(dataset_name, specific_dataset, critic_checkpoint)


        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        self.config.critic_d_s = critic_obs_dim
        if(not self.config.explore):
              self.config.gamma = 0.0

    @staticmethod
    def _make_state(model_def, torch_state_dict):
        # TODO(checkpoint-bridge): see TotalReward._make_state — torch state_dict -> flax param tree remap.
        return TrainState.create(model_def, params=torch_state_dict, tx=None)

    def get_beta(self):
        return self.config.beta

    def sigmoid(self, s, a, s_next):
        total = jnp.array([0.0])
        for i in range(len(self.kernels)):
            mu, log_std = self.kernels[i](s, a)
            #lp = self.kernels[i].log_prob(s_next, mu, log_std)
            lp = self.kernels[i](s_next, mu, log_std, method='log_prob')
            total = total + lp
        avg = total / len(self.kernels)
        x =  self.config.min_log_prob - avg
        c = _softplus_beta(x, self.config.beta)
        return c


    def reward_processor(self, s):
        s_n = np.asarray(s)
        s_n = self.reward_stat.norm_obs(s_n)
        s = jnp.asarray(s_n, dtype=jnp.float32)
        return s

    def kernel_processor(self, s):
        s_n = np.asarray(s)
        s_n = self.kernel_stat.norm_obs(s_n)
        s = jnp.asarray(s_n, dtype=jnp.float32)
        return s

    def critic_processor(self, s):
        s_n = np.asarray(s)
        s_n = self.critic_stat.norm_obs(s_n)
        s = jnp.asarray(s_n, dtype=jnp.float32)
        return s

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
            C += c.squeeze(0)
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
            total_reward += (r.squeeze(0)) - (lam  *  c.squeeze(0))

        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        r = self.reward_net(s_norm_reward, a)
        final_s_critic = x[H-1][:self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic)[None]
        v = self.critic(final_s_norm_critic)
        total_reward +=   (r.squeeze(0)) + ( (self.config.critic_gamma**(H-1)) * v.squeeze(0))
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


            def reward_fn(sr, av):
                return self.reward_net(sr, av)
            r, reward_vjp = jax.vjp(reward_fn, s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)

            grads = reward_vjp(jnp.ones_like(r))
            r_s = grads[0].squeeze(0) * jnp.asarray(
                (1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
            r_a = grads[1].squeeze(0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)



            def constraint_fn(sk, av, snk):
                return self.sigmoid(sk, av, snk)
            c, constraint_vjp = jax.vjp(constraint_fn, s_norm_kernel, a, s_next_norm_kernel)
            grads = constraint_vjp(jnp.ones_like(c))
            c_s = grads[0].squeeze(0) * jnp.asarray(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   dtype=jnp.float32)
            c_a = grads[1].squeeze(0)
            c_s_next = grads[2].squeeze(0) * jnp.asarray(
                1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor), dtype=jnp.float32)
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient +=  ((r_s_grad + r_a_grad)) - (lam * (c_s_grad + c_a_grad + c_s_next_grad))

            total_reward += (r.squeeze(0)) - (lam  * ( c.squeeze(0)))
            #total_reward += (1/H)*(r.squeeze(0)) - lam * (1/(H-1)) * (c.squeeze(0) - self.config.delta)



        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        def reward_fn(sr, av):
            return self.reward_net(sr, av)
        r, reward_vjp = jax.vjp(reward_fn, s_norm_reward, a)


        grads = reward_vjp(jnp.ones_like(r))
        r_s = grads[0].squeeze(0) * jnp.asarray((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)),
                                                dtype=jnp.float32)
        r_a = grads[1].squeeze(0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H-1)

        final_s_critic = x[H-1][:self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic)[None]
        def critic_fn(sc):
            return self.critic(sc)
        v, critic_vjp = jax.vjp(critic_fn, final_s_norm_critic)
        grads = critic_vjp(jnp.ones_like(v))
        v_s = grads[0].squeeze(0) * jnp.asarray((1/np.maximum(self.critic_stat.obs_std, self.critic_stat.std_floor)),
                                                dtype=jnp.float32)
        grad_critic = self.makeGrad_Critic(H, v_s, H-1)


        #gradient += ((r_s_grad + r_a_grad))  + ( (self.config.critic_gamma**(H-1)) * grad_critic)
        #total_reward +=  (r.squeeze(0)) + ((self.config.critic_gamma**(H-1)) * v.squeeze(0))
        gradient +=   ( (self.config.critic_gamma**(H-1)) * grad_critic)
        total_reward +=   ((self.config.critic_gamma**(H-1)) * v.squeeze(0))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient
