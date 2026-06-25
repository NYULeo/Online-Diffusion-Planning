'''Trajectory reward (reward - lambda * constraint) with input-gradients, MoG/robust kernel + optional critic.

JAX/Flax port (FQL-style) of the torch original. The reward/kernel/critic networks are *frozen pretrained*
models loaded from existing torch checkpoints; this module holds them as (model_def, params) pairs and
computes the per-step reward together with the gradient of that reward w.r.t. the trajectory input `x`
(an input-gradient via jax.vjp, see CONVERSION_GUIDE.md section 7).
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
from Pretrain.Transition_Kernel.Kernel_Backbone import (
    compute_log_density, compute_log_density_mog, compute_total_mahalanobis_score_mog,
)
from Pretrain.Critic.nets import Critic
from Finetuning.utils import (
    get_reward_model, get_kernel, get_reward_stats, get_kernel_stats, get_critic_model, get_critic_stats,
)


def _softplus_beta(x, beta):
    '''PyTorch-faithful softplus with beta: (1/beta) * log(1 + exp(beta * x)).'''
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
        # TODO(checkpoint-bridge): torch did
        #   self.reward_net = SimpleReward(obs_dim, act_dim, hidden_dim_reward, num_hidden_layers_reward); load_state_dict(reward_state_dict)
        # Here we keep the frozen pretrained net as a (model_def, params) pair; reward_state_dict is the
        # ingested torch checkpoint mapped to a flax param tree by the SimpleReward loader.
        self.reward_net = SimpleReward(obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward)
        self.reward_params = reward_state_dict
        self.kernels = []
        self.kernel_params = []
        self.config.delta = _softplus_beta(jnp.array(0.0), beta=self.config.beta)
        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        if self.config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                kernel_net = RobustTransitionKernel(
                    obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel
                )
                self.kernels.append(kernel_net)
                self.kernel_params.append(sd)  # TODO(checkpoint-bridge): torch kernel_net.load_state_dict(sd)
        else:
            for sd in kernel_state_dicts:
                kernel_net = MoGTransitionKernel(
                    obs_dim, act_dim, self.config.kernel_num_modes,
                    self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel,
                    noise_floor=self.config.kernel_noise_floor
                )
                self.kernels.append(kernel_net)
                self.kernel_params.append(sd)  # TODO(checkpoint-bridge): torch kernel_net.load_state_dict(sd)
        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)

        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)


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
                mu, log_std = self.kernels[i].apply({'params': self.kernel_params[i]}, s, a)
                lp = self.kernels[i].apply({'params': self.kernel_params[i]}, s_next, mu, log_std, method='log_prob')
                total = total + lp
            avg = total / len(self.kernels)
        else:
            # §11: compute_log_density_mog iterates `for model_def, params in kernels`; this class keeps
            # defs and params in parallel lists, so pair them into (model_def, params) tuples here.
            avg = compute_log_density_mog(list(zip(self.kernels, self.kernel_params)), s, a, s_next)
        x = self.config.min_log_prob - avg
        c = _softplus_beta(x, beta=self.config.beta)
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
           S_next = S_next.at[i+1, :self.config.d_s].set(s_next_grad)
           return S, A, S_next
        return S, A

    def get_c(self, x):
        H, D = x.shape
        C = jnp.array(0.0)
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
        total_reward = jnp.array(0.0)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s)[None]
            a = x[i][self.config.d_s:][None]


            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]


            r = self.reward_net.apply({'params': self.reward_params}, s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward += (1/H)*(jnp.squeeze(r, 0)) - lam  * ( (1/(H-1)) * jnp.squeeze(c, 0))

        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        r = self.reward_net.apply({'params': self.reward_params}, s_norm_reward, a)
        total_reward +=  (1/H) * (jnp.squeeze(r, 0))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward

    def forward(self, x: jnp.ndarray, lam: float):
        H, D = x.shape
        total_reward = jnp.array(0.0)
        gradient = jnp.zeros((H, D))
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s)[None]
            a = x[i][self.config.d_s:][None]


            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]


            # input-gradient of r w.r.t. (s_norm_reward, a) via vjp (section 7); frozen net, no param grads.
            def _r_fn(s_in, a_in):
                return self.reward_net.apply({'params': self.reward_params}, s_in, a_in)
            r, r_vjp = jax.vjp(_r_fn, s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)

            grads = r_vjp(jnp.ones_like(r))
            r_s = jnp.squeeze(grads[0], 0) * jnp.asarray((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
            r_a = jnp.squeeze(grads[1], 0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)



            # input-gradient of c w.r.t. (s_norm_kernel, a, s_next_norm_kernel) via vjp (section 7).
            def _c_fn(s_in, a_in, s_next_in):
                return self.sigmoid(s_in, a_in, s_next_in)
            c_val, c_vjp = jax.vjp(_c_fn, s_norm_kernel, a, s_next_norm_kernel)
            grads = c_vjp(jnp.ones_like(c_val))
            c_s = jnp.squeeze(grads[0], 0) * jnp.asarray(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   dtype=jnp.float32)
            c_a = jnp.squeeze(grads[1], 0)
            c_s_next = jnp.squeeze(grads[2], 0) * jnp.asarray(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   dtype=jnp.float32)
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient +=  (1/H)*((r_s_grad + r_a_grad)) - lam * (1/(H-1)) * (c_s_grad + c_a_grad + c_s_next_grad)

            total_reward += (1/H)*(jnp.squeeze(r, 0)) - lam  * ( (1/(H-1)) * jnp.squeeze(c, 0))
            #total_reward += (1/H)*(r.squeeze(0)) - lam * (1/(H-1)) * (c.squeeze(0) - self.config.delta)



        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        def _r_fn(s_in, a_in):
            return self.reward_net.apply({'params': self.reward_params}, s_in, a_in)
        r, r_vjp = jax.vjp(_r_fn, s_norm_reward, a)


        grads = r_vjp(jnp.ones_like(r))
        r_s = jnp.squeeze(grads[0], 0) * jnp.asarray((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
        r_a = jnp.squeeze(grads[1], 0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H-1)



        gradient += (1/H) * ((r_s_grad + r_a_grad))
        total_reward +=  (1/H) * (jnp.squeeze(r, 0))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient


class TotalReward_Critic:
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int, critic_checkpoint: int, task_id: Optional[int] = None):
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint, task_id)
        self.config.device = device
        # TODO(checkpoint-bridge): torch reward_net.load_state_dict(reward_state_dict); kept as (model_def, params).
        self.reward_net = SimpleReward(obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward)
        self.reward_params = reward_state_dict
        self.kernels = []
        self.kernel_params = []
        self.config.delta = _softplus_beta(jnp.array(0.0), beta=self.config.beta)


        critic_state_dict, critic_obs_dim = get_critic_model(dataset_name, specific_dataset, task_id, critic_checkpoint)
        # TODO(checkpoint-bridge): torch critic.load_state_dict(critic_state_dict); kept as (model_def, params).
        self.critic = Critic(critic_obs_dim, self.config.hidden_dim_critic, self.config.num_hidden_layers_critic)
        self.critic_params = critic_state_dict

        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        if self.config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                kernel_net = RobustTransitionKernel(
                    obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel
                )
                self.kernels.append(kernel_net)
                self.kernel_params.append(sd)  # TODO(checkpoint-bridge): torch kernel_net.load_state_dict(sd)
        else:
            for sd in kernel_state_dicts:
                kernel_net = MoGTransitionKernel(
                    obs_dim, act_dim, self.config.kernel_num_modes,
                    self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel,
                    noise_floor=self.config.kernel_noise_floor
                )
                self.kernels.append(kernel_net)
                self.kernel_params.append(sd)  # TODO(checkpoint-bridge): torch kernel_net.load_state_dict(sd)
        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)
        self.critic_stat = get_critic_stats(dataset_name, specific_dataset, task_id, critic_checkpoint)


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
                mu, log_std = self.kernels[i].apply({'params': self.kernel_params[i]}, s, a)
                lp = self.kernels[i].apply({'params': self.kernel_params[i]}, s_next, mu, log_std, method='log_prob')
                total = total + lp
            avg = total / len(self.kernels)
        else:
            # §11: compute_log_density_mog iterates `for model_def, params in kernels`; this class keeps
            # defs and params in parallel lists, so pair them into (model_def, params) tuples here.
            avg = compute_log_density_mog(list(zip(self.kernels, self.kernel_params)), s, a, s_next)
        x = self.config.min_log_prob - avg
        c = _softplus_beta(x, beta=self.config.beta)
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
           S_next = S_next.at[i+1, :self.config.d_s].set(s_next_grad)
           return S, A, S_next
        return S, A

    def makeGrad_Critic(self, H, s_grad, i):
        S = jnp.zeros((H, (self.config.d_s + self.config.d_a)))
        S = S.at[i, :self.config.critic_d_s].set(s_grad)
        return S

    def get_c(self, x):
        H, D = x.shape
        C = jnp.array(0.0)
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
        total_reward = jnp.array(0.0)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s)[None]
            a = x[i][self.config.d_s:][None]


            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]


            r = self.reward_net.apply({'params': self.reward_params}, s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward += (jnp.squeeze(r, 0)) - (lam  *  jnp.squeeze(c, 0))

        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        r = self.reward_net.apply({'params': self.reward_params}, s_norm_reward, a)
        final_s_critic = x[H-1][:self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic)[None]
        v = self.critic.apply({'params': self.critic_params}, final_s_norm_critic)
        total_reward +=   (jnp.squeeze(r, 0)) + ( (self.config.critic_gamma**(H-1)) * jnp.squeeze(v, 0))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward

    def forward(self, x: jnp.ndarray, lam: float):
        H, D = x.shape
        total_reward = jnp.array(0.0)
        gradient = jnp.zeros((H, D))
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s)[None]
            a = x[i][self.config.d_s:][None]


            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s)[None]
            s_next_norm_kernel = self.kernel_processor(s_next)[None]


            # input-gradient of r w.r.t. (s_norm_reward, a) via vjp (section 7).
            def _r_fn(s_in, a_in):
                return self.reward_net.apply({'params': self.reward_params}, s_in, a_in)
            r, r_vjp = jax.vjp(_r_fn, s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)

            grads = r_vjp(jnp.ones_like(r))
            r_s = jnp.squeeze(grads[0], 0) * jnp.asarray((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
            r_a = jnp.squeeze(grads[1], 0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)



            # input-gradient of c w.r.t. (s_norm_kernel, a, s_next_norm_kernel) via vjp (section 7).
            def _c_fn(s_in, a_in, s_next_in):
                return self.sigmoid(s_in, a_in, s_next_in)
            c_val, c_vjp = jax.vjp(_c_fn, s_norm_kernel, a, s_next_norm_kernel)
            grads = c_vjp(jnp.ones_like(c_val))
            c_s = jnp.squeeze(grads[0], 0) * jnp.asarray(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   dtype=jnp.float32)
            c_a = jnp.squeeze(grads[1], 0)
            c_s_next = jnp.squeeze(grads[2], 0) * jnp.asarray(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   dtype=jnp.float32)
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)

            gradient +=  ((r_s_grad + r_a_grad)) - (lam * (c_s_grad + c_a_grad + c_s_next_grad))

            total_reward += (jnp.squeeze(r, 0)) - (lam  * ( jnp.squeeze(c, 0)))
            #total_reward += (1/H)*(r.squeeze(0)) - lam * (1/(H-1)) * (c.squeeze(0) - self.config.delta)



        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s)[None]
        a = x[H-1][self.config.d_s:][None]
        def _r_fn(s_in, a_in):
            return self.reward_net.apply({'params': self.reward_params}, s_in, a_in)
        r, r_vjp = jax.vjp(_r_fn, s_norm_reward, a)


        grads = r_vjp(jnp.ones_like(r))
        r_s = jnp.squeeze(grads[0], 0) * jnp.asarray((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), dtype=jnp.float32)
        r_a = jnp.squeeze(grads[1], 0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H-1)

        final_s_critic = x[H-1][:self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic)[None]
        # input-gradient of v w.r.t. final_s_norm_critic via vjp (section 7).
        def _v_fn(s_in):
            return self.critic.apply({'params': self.critic_params}, s_in)
        v, v_vjp = jax.vjp(_v_fn, final_s_norm_critic)
        grads = v_vjp(jnp.ones_like(v))
        v_s = jnp.squeeze(grads[0], 0) * jnp.asarray((1/np.maximum(self.critic_stat.obs_std, self.critic_stat.std_floor)), dtype=jnp.float32)
        grad_critic = self.makeGrad_Critic(H, v_s, H-1)


        #gradient += ((r_s_grad + r_a_grad))  + ( (self.config.critic_gamma**(H-1)) * grad_critic)
        #total_reward +=  (r.squeeze(0)) + ((self.config.critic_gamma**(H-1)) * v.squeeze(0))
        gradient +=   ( (self.config.critic_gamma**(H-1)) * grad_critic)
        total_reward +=   ((self.config.critic_gamma**(H-1)) * jnp.squeeze(v, 0))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient
