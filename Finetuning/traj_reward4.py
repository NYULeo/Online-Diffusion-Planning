import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import torch
import torch.nn as nn
from Pretrain.Rewards.nets import SimpleReward
from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel, MoGTransitionKernel
from Pretrain.Transition_Kernel.Kernel_Backbone import compute_log_density, compute_log_density_mog
from Pretrain.Critic.nets import Critic
from Finetuning.utils import get_reward_model, get_kernel, get_reward_stats, get_kernel_stats, get_critic_model, get_critic_stats, get_Q_stats
from typing import Optional
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np



@dataclass
class RewardConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    beta: float
    min_log_prob: float
    quantile: float = 0.999
    number_of_generated_plans: int = 50
    explore: bool = True
    gamma: float = 0.8
    gae_lam: float = 0.95
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
    

class TotalReward(nn.Module):
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
        super().__init__()
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint, task_id)
        self.config.device = device
        self.reward_net = SimpleReward(obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()
        self.kernels = []
        self.config.delta = F.softplus(torch.tensor(0.0, requires_grad = False), beta = self.config.beta).to(self.config.device)
        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        if self.config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                kernel_net = RobustTransitionKernel(
                    obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel
                ).to(self.config.device)
                kernel_net.load_state_dict(sd)
                kernel_net.eval()
                self.kernels.append(kernel_net)
        else:
            for sd in kernel_state_dicts:
                kernel_net = MoGTransitionKernel(
                    obs_dim, act_dim, self.config.kernel_num_modes,
                    self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel,
                    noise_floor=self.config.kernel_noise_floor
                ).to(self.config.device)
                kernel_net.load_state_dict(sd)
                kernel_net.eval()
                self.kernels.append(kernel_net)
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
            total = torch.tensor([0.0], device=self.config.device, requires_grad=True)
            for i in range(len(self.kernels)):
                mu, log_std = self.kernels[i](s, a)
                lp = self.kernels[i].log_prob(s_next, mu, log_std)
                total = total + lp
            avg = total / len(self.kernels)
        else:
            avg = compute_log_density_mog(self.kernels, s, a, s_next)
        x = self.config.min_log_prob - avg
        c = F.softplus(x, beta=self.config.beta)
        return c
    

    def reward_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.reward_stat.norm_obs(s_n)
        s = torch.tensor(s_n, dtype = torch.float32, device = self.config.device, requires_grad = True)
        return s
    
    def kernel_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.kernel_stat.norm_obs(s_n)
        s = torch.tensor(s_n, dtype = torch.float32, device = self.config.device, requires_grad = True)
        return s

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[torch.Tensor] = None):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
        A = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
        S[i][:self.config.d_s] = s_grad
        A[i][self.config.d_s:] = a_grad
        if s_next_grad is not None:
           S_next = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
           S_next[i+1][:self.config.d_s] = s_next_grad
           return S, A, S_next
        return S, A
        
    def get_c(self, x):
        H, D = x.shape
        C = torch.tensor(0.0, device = self.config.device, requires_grad=False)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            a = x[i][self.config.d_s:].unsqueeze(0)
            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            C += c.squeeze(0)
        C = C / (H-1)
        C = C - self.config.delta
        return C

    def predict(self, x: torch.Tensor, lam: float):
        H, D = x.shape
        total_reward = torch.tensor(0.0, device = self.config.device, requires_grad = False)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(False).to(self.config.device)
            a = x[i][self.config.d_s:].unsqueeze(0).requires_grad_(False).to(self.config.device)
            
           
            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0).requires_grad_(False).to(self.config.device)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0).requires_grad_(False).to(self.config.device)
 
           
            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward += (1/H)*((self.config.critic_gamma**i)*(r.squeeze(0))) - lam  * ( (1/(H-1)) * c.squeeze(0))
        
        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(False)
        a = x[H-1][self.config.d_s:].unsqueeze(0).requires_grad_(False)
        r = self.reward_net(s_norm_reward, a)
        total_reward +=  (1/H) * ((self.config.critic_gamma**(H-1))*(r.squeeze(0)))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward

    def forward(self, x: torch.Tensor, lam: float):
        H, D = x.shape
        total_reward = torch.tensor(0.0, device=self.config.device, requires_grad = False)
        gradient = torch.zeros(H, D, device = self.config.device, requires_grad = False)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True).to(self.config.device)
            a = x[i][self.config.d_s:].unsqueeze(0).requires_grad_(True).to(self.config.device)
            
           
            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0).requires_grad_(True).to(self.config.device)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0).requires_grad_(True).to(self.config.device)
 
           
            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
           
            grads = torch.autograd.grad(
                        outputs = r,
                        inputs = (s_norm_reward, a),
                        grad_outputs = torch.ones_like(r),
                        create_graph = False,
                        retain_graph = False,
                        allow_unused = False
                    )
            r_s = grads[0].squeeze(0) * torch.tensor((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), device = self.config.device, dtype=torch.float32, requires_grad = False)
            r_a = grads[1].squeeze(0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)
            
            
            
            grads = torch.autograd.grad(
                        outputs = c,
                        inputs = (s_norm_kernel, a, s_next_norm_kernel),
                        grad_outputs = torch.ones_like(c),
                        create_graph = True,
                        retain_graph = True
                        
                    )
            c_s = grads[0].squeeze(0) * torch.tensor(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   device = self.config.device, dtype=torch.float32, requires_grad = False)
            c_a = grads[1].squeeze(0)   
            c_s_next = grads[2].squeeze(0) * torch.tensor(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   device = self.config.device, dtype=torch.float32, requires_grad = False)
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)
            
            gradient +=  (1/H)*((self.config.critic_gamma**i)*(r_s_grad + r_a_grad)) - lam * (1/(H-1)) * (c_s_grad + c_a_grad + c_s_next_grad)
            
            total_reward += (1/H)*((self.config.critic_gamma**i)*(r.squeeze(0))) - lam  * ( (1/(H-1)) * c.squeeze(0))
            #total_reward += (1/H)*(r.squeeze(0)) - lam * (1/(H-1)) * (c.squeeze(0) - self.config.delta)
            
        

        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
        a = x[H-1][self.config.d_s:].unsqueeze(0).requires_grad_(True)
        r = self.reward_net(s_norm_reward, a)
        

        grads = torch.autograd.grad(
                        outputs = r,
                        inputs = (s_norm_reward, a),
                        grad_outputs = torch.ones_like(r),
                        create_graph = False,
                        retain_graph = False
                )
        r_s = grads[0].squeeze(0) * torch.tensor((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), device = self.config.device, dtype=torch.float32, requires_grad = False)
        r_a = grads[1].squeeze(0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H-1)

       
       
        gradient += (1/H) * ((self.config.critic_gamma**(H-1))*(r_s_grad + r_a_grad)) 
        total_reward +=  (1/H) * ((self.config.critic_gamma**(H-1))*(r.squeeze(0)))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient

"""
class TotalReward_Critic(nn.Module):
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int, critic_checkpoint: int, task_id: Optional[int] = None):
        super().__init__()
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint, task_id)
        self.config.device = device
        self.reward_net = SimpleReward(obs_dim, act_dim, self.config.hidden_dim_reward, self.config.num_hidden_layers_reward).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()
        self.kernels = []
        self.config.delta = F.softplus(torch.tensor(0.0, requires_grad = False), beta = self.config.beta).to(self.config.device)


        critic_state_dict, critic_obs_dim = get_critic_model(dataset_name, specific_dataset, task_id, critic_checkpoint)
        self.critic = Critic(critic_obs_dim, self.config.hidden_dim_critic, self.config.num_hidden_layers_critic).to(self.config.device)
        self.critic.load_state_dict(critic_state_dict)
        self.critic.eval()

        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        if self.config.type_kernel == 'robust':
            for sd in kernel_state_dicts:
                kernel_net = RobustTransitionKernel(
                    obs_dim, act_dim, self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel
                ).to(self.config.device)
                kernel_net.load_state_dict(sd)
                kernel_net.eval()
                self.kernels.append(kernel_net)
        else:
            for sd in kernel_state_dicts:
                kernel_net = MoGTransitionKernel(
                    obs_dim, act_dim, self.config.kernel_num_modes,
                    self.config.num_hidden_layers_kernel, self.config.hidden_dim_kernel,
                    noise_floor=self.config.kernel_noise_floor
                ).to(self.config.device)
                kernel_net.load_state_dict(sd)
                kernel_net.eval()
                self.kernels.append(kernel_net)
        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint, task_id)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)
        self.critic_stat = get_critic_stats(dataset_name, specific_dataset, task_id, 0)
        self.q_stats = get_Q_stats(dataset_name, specific_dataset, task_id, critic_checkpoint)
       

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        self.config.critic_d_s = critic_obs_dim
        if(not self.config.explore):
              self.config.gamma = 0.0
        
    def get_beta(self):
        return self.config.beta

    def sigmoid(self, s, a, s_next):
        if self.config.type_kernel == 'robust':
            total = torch.tensor([0.0], device=self.config.device, requires_grad=True)
            for i in range(len(self.kernels)):
                mu, log_std = self.kernels[i](s, a)
                lp = self.kernels[i].log_prob(s_next, mu, log_std)
                total = total + lp
            avg = total / len(self.kernels)
        else:
            avg = compute_log_density_mog(self.kernels, s, a, s_next)
        x = self.config.min_log_prob - avg
        c = F.softplus(x, beta=self.config.beta)
        return c
    
    def reward_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.reward_stat.norm_obs(s_n)
        s = torch.tensor(s_n, dtype = torch.float32, device = self.config.device, requires_grad = True)
        return s
    
    def kernel_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.kernel_stat.norm_obs(s_n)
        s = torch.tensor(s_n, dtype = torch.float32, device = self.config.device, requires_grad = True)
        return s
    
    def critic_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.critic_stat.norm_obs(s_n)
        s = torch.tensor(s_n, dtype = torch.float32, device = self.config.device, requires_grad = True)
        return s

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[torch.Tensor] = None):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
        A = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
        S[i][:self.config.d_s] = s_grad
        A[i][self.config.d_s:] = a_grad
        if s_next_grad is not None:
           S_next = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
           S_next[i+1][:self.config.d_s] = s_next_grad
           return S, A, S_next
        return S, A
    
    def makeGrad_Critic(self, H, s_grad, i):
        S = torch.zeros(H, (self.config.d_s + self.config.d_a), device = self.config.device)
        S[i][:self.config.critic_d_s] = s_grad
        return S

    def get_c(self, x):
        H, D = x.shape
        C = torch.tensor(0.0, device = self.config.device, requires_grad=False)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            a = x[i][self.config.d_s:].unsqueeze(0)
            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            C += c.squeeze(0)
        C = C / (H-1)
        C = C - self.config.delta
        return C

    def predict(self, x: torch.Tensor, lam: float):
        H, D = x.shape
        total_reward = torch.tensor(0.0, device = self.config.device, requires_grad = False)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(False).to(self.config.device)
            a = x[i][self.config.d_s:].unsqueeze(0).requires_grad_(False).to(self.config.device)
            
           
            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0).requires_grad_(False).to(self.config.device)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0).requires_grad_(False).to(self.config.device)
 
           
            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
            total_reward += ((self.config.critic_gamma**i)*(r.squeeze(0))) - (lam  *  c.squeeze(0))
        
        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(False)
        a = x[H-1][self.config.d_s:].unsqueeze(0).requires_grad_(False)
        r = self.reward_net(s_norm_reward, a)
        final_s_critic = x[H-1][:self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic).unsqueeze(0).requires_grad_(False)
        v = self.critic(final_s_norm_critic)
        #total_reward +=   ((self.config.critic_gamma**(H-1))*(r.squeeze(0))) + ( (self.config.critic_gamma**(H-1)) * v.squeeze(0))
        total_reward +=   ( (self.config.critic_gamma**(H-1)) *  (  (self.q_stats.Q_std * v.squeeze(0)) + self.q_stats.Q_mean  ))
        total_reward = total_reward + (lam  * self.config.delta) 
        return total_reward

    def forward(self, x: torch.Tensor, lam: float):
        H, D = x.shape
        total_reward = torch.tensor(0.0, device=self.config.device, requires_grad = False)
        gradient = torch.zeros(H, D, device = self.config.device, requires_grad = False)
        for i in range(H-1):
            s = x[i][:self.config.d_s]
            s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True).to(self.config.device)
            a = x[i][self.config.d_s:].unsqueeze(0).requires_grad_(True).to(self.config.device)
            
           
            s_next = x[i+1][:self.config.d_s]
            s_norm_kernel = self.kernel_processor(s).unsqueeze(0).requires_grad_(True).to(self.config.device)
            s_next_norm_kernel = self.kernel_processor(s_next).unsqueeze(0).requires_grad_(True).to(self.config.device)
 
           
            r = self.reward_net(s_norm_reward, a)
            c = self.sigmoid(s_norm_kernel, a, s_next_norm_kernel)
           
            grads = torch.autograd.grad(
                        outputs = r,
                        inputs = (s_norm_reward, a),
                        grad_outputs = torch.ones_like(r),
                        create_graph = False,
                        retain_graph = False,
                        allow_unused = False
                    )
            r_s = grads[0].squeeze(0) * torch.tensor((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), device = self.config.device, dtype=torch.float32, requires_grad = False)
            r_a = grads[1].squeeze(0)
            r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, i)
            
            
            
            grads = torch.autograd.grad(
                        outputs = c,
                        inputs = (s_norm_kernel, a, s_next_norm_kernel),
                        grad_outputs = torch.ones_like(c),
                        create_graph = True,
                        retain_graph = True
                        
                    )
            c_s = grads[0].squeeze(0) * torch.tensor(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   device = self.config.device, dtype=torch.float32, requires_grad = False)
            c_a = grads[1].squeeze(0)   
            c_s_next = grads[2].squeeze(0) * torch.tensor(1/np.maximum(self.kernel_stat.obs_std, self.kernel_stat.std_floor),
                                                   device = self.config.device, dtype=torch.float32, requires_grad = False)
            c_s_grad, c_a_grad, c_s_next_grad = self.makeGrad(H, c_s, c_a, i, c_s_next)
            
            gradient +=  ((self.config.critic_gamma**i)*(r_s_grad + r_a_grad)) - (lam * (c_s_grad + c_a_grad + c_s_next_grad))
            
            total_reward += ((self.config.critic_gamma**i)*(r.squeeze(0))) - (lam  * ( c.squeeze(0)))
            #total_reward += (1/H)*(r.squeeze(0)) - lam * (1/(H-1)) * (c.squeeze(0) - self.config.delta)
            
        

        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(True)
        a = x[H-1][self.config.d_s:].unsqueeze(0).requires_grad_(True)
        r = self.reward_net(s_norm_reward, a)
        

        grads = torch.autograd.grad(
                        outputs = r,
                        inputs = (s_norm_reward, a),
                        grad_outputs = torch.ones_like(r),
                        create_graph = False,
                        retain_graph = False
                )
        r_s = grads[0].squeeze(0) * torch.tensor((1/np.maximum(self.reward_stat.obs_std, self.reward_stat.std_floor)), device = self.config.device, dtype=torch.float32, requires_grad = False)
        r_a = grads[1].squeeze(0)
        r_s_grad, r_a_grad = self.makeGrad(H, r_s, r_a, H-1)
        
        final_s_critic = x[H-1][:self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic).unsqueeze(0).requires_grad_(True)
        v = self.critic(final_s_norm_critic)
        grads = torch.autograd.grad(
                outputs = v,
                inputs = (final_s_norm_critic),
                grad_outputs = torch.ones_like(v),
                create_graph = False,
                retain_graph = False
            ) 
        v_s = grads[0].squeeze(0) * torch.tensor((1/np.maximum(self.critic_stat.obs_std, self.critic_stat.std_floor)), device = self.config.device, dtype=torch.float32, requires_grad = False)
        grad_critic = self.makeGrad_Critic(H, v_s, H-1)
        
       
        #gradient += ((r_s_grad + r_a_grad))  + ( (self.config.critic_gamma**(H-1)) * grad_critic)
        #total_reward +=  (r.squeeze(0)) + ((self.config.critic_gamma**(H-1)) * v.squeeze(0))
        gradient +=   ( (self.config.critic_gamma**(H-1)) *  (self.q_stats.Q_std * grad_critic))
        total_reward +=   ((self.config.critic_gamma**(H-1)) * (  (self.q_stats.Q_std * v.squeeze(0)) + self.q_stats.Q_mean  )  )
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient

"""




class TotalReward_Critic(nn.Module):
    def __init__(
        self,
        device,
        config: RewardConfig,
        dataset_name: str,
        specific_dataset: str,
        reward_checkpoint: int,
        kernel_checkpoint: int,
        critic_checkpoint: int,
        task_id: Optional[int] = None,
    ):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------ reward
        reward_state_dict, obs_dim, act_dim = get_reward_model(
            dataset_name, specific_dataset, reward_checkpoint, task_id
        )
        self.config.device = device
        self.reward_net = SimpleReward(
            obs_dim,
            act_dim,
            self.config.hidden_dim_reward,
            self.config.num_hidden_layers_reward,
        ).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()

        # ------------------------------------------------------------------ kernels
        self.kernels = []
        self.config.delta = F.softplus(
            torch.tensor(0.0, requires_grad=False), beta=self.config.beta
        ).to(self.config.device)

        kernel_state_dicts, obs_dim, act_dim = get_kernel(
            dataset_name, specific_dataset, kernel_checkpoint
        )
        if self.config.type_kernel == "robust":
            for sd in kernel_state_dicts:
                kernel_net = RobustTransitionKernel(
                    obs_dim,
                    act_dim,
                    self.config.num_hidden_layers_kernel,
                    self.config.hidden_dim_kernel,
                ).to(self.config.device)
                kernel_net.load_state_dict(sd)
                kernel_net.eval()
                self.kernels.append(kernel_net)
        else:
            for sd in kernel_state_dicts:
                kernel_net = MoGTransitionKernel(
                    obs_dim,
                    act_dim,
                    self.config.kernel_num_modes,
                    self.config.num_hidden_layers_kernel,
                    self.config.hidden_dim_kernel,
                    noise_floor=self.config.kernel_noise_floor,
                ).to(self.config.device)
                kernel_net.load_state_dict(sd)
                kernel_net.eval()
                self.kernels.append(kernel_net)

        # ------------------------------------------------------------------ critic
        critic_state_dict, critic_obs_dim = get_critic_model(
            dataset_name, specific_dataset, task_id, critic_checkpoint
        )
        self.critic = Critic(
            critic_obs_dim,
            self.config.hidden_dim_critic,
            self.config.num_hidden_layers_critic,
        ).to(self.config.device)
        self.critic.load_state_dict(critic_state_dict)
        self.critic.eval()

        # ------------------------------------------------------------------ stats
        self.reward_stat = get_reward_stats(
            dataset_name, specific_dataset, reward_checkpoint, task_id
        )
        self.kernel_stat = get_kernel_stats(
            dataset_name, specific_dataset, kernel_checkpoint
        )
        self.critic_stat = get_critic_stats(
            dataset_name, specific_dataset, task_id, 0
        )
        self.q_stats = get_Q_stats(
            dataset_name, specific_dataset, task_id, critic_checkpoint
        )

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        self.config.critic_d_s = critic_obs_dim

        if not self.config.explore:
            self.config.gamma = 0.0

    # ---------------------------------------------------------------------- helpers
    def get_beta(self):
        return self.config.beta

    def sigmoid(self, s, a, s_next):
        if self.config.type_kernel == "robust":
            total = torch.tensor([0.0], device=self.config.device, requires_grad=True)
            for i in range(len(self.kernels)):
                mu, log_std = self.kernels[i](s, a)
                lp = self.kernels[i].log_prob(s_next, mu, log_std)
                total = total + lp
            avg = total / len(self.kernels)
        else:
            avg = compute_log_density_mog(self.kernels, s, a, s_next)
        x = self.config.min_log_prob - avg
        c = F.softplus(x, beta=self.config.beta)
        return c

    def reward_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.reward_stat.norm_obs(s_n)
        return torch.tensor(
            s_n, dtype=torch.float32, device=self.config.device, requires_grad=True
        )

    def kernel_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.kernel_stat.norm_obs(s_n)
        return torch.tensor(
            s_n, dtype=torch.float32, device=self.config.device, requires_grad=True
        )

    def critic_processor(self, s):
        s_n = s.detach().cpu().numpy()
        s_n = self.critic_stat.norm_obs(s_n)
        return torch.tensor(
            s_n, dtype=torch.float32, device=self.config.device, requires_grad=True
        )

    def makeGrad(self, H, s_grad, a_grad, i, s_next_grad: Optional[torch.Tensor] = None):
        S = torch.zeros(H, self.config.d_s + self.config.d_a, device=self.config.device)
        A = torch.zeros(H, self.config.d_s + self.config.d_a, device=self.config.device)
        S[i, : self.config.d_s] = s_grad
        A[i, self.config.d_s :] = a_grad
        if s_next_grad is not None:
            S_next = torch.zeros(
                H, self.config.d_s + self.config.d_a, device=self.config.device
            )
            S_next[i + 1, : self.config.d_s] = s_next_grad
            return S, A, S_next
        return S, A

    def makeGrad_Critic(self, H, s_grad, i):
        S = torch.zeros(H, self.config.d_s + self.config.d_a, device=self.config.device)
        S[i, : self.config.critic_d_s] = s_grad
        return S

    def get_c(self, x):
        H, _ = x.shape
        C = torch.tensor(0.0, device=self.config.device)
        for i in range(H - 1):
            s = x[i, : self.config.d_s]
            a = x[i, self.config.d_s :].unsqueeze(0)
            s_next = x[i + 1, : self.config.d_s]
            s_k = self.kernel_processor(s).unsqueeze(0)
            s_next_k = self.kernel_processor(s_next).unsqueeze(0)
            c = self.sigmoid(s_k, a, s_next_k)
            C = C + c.squeeze(0)
        C = C / (H - 1) - self.config.delta
        return C

    # ---------------------------------------------------------------------- core
    def _compute_gae_style_return(
        self, x: torch.Tensor, lam: float,  with_grad: bool = False
    ):
        
        H, D = x.shape
        device = self.config.device
        gamma = self.config.critic_gamma
        n = H

        # -------------------- pre-compute rewards and critic values --------------------
        rs = []          # r[0] ... r[n-2]
        r_grads = []     # (s_grad, a_grad) or None
        vs = []          # denormalised V[0] ... V[n-1]
        v_grads = []     # s_grad or None

        for i in range(n):
            s = x[i, : self.config.d_s]

            # reward (only needed for i = 0 ... n-2)
            if i < n - 1:
                a = x[i, self.config.d_s :]
                s_r = self.reward_processor(s).unsqueeze(0)
                a_t = a.unsqueeze(0)
                if with_grad:
                    s_r = s_r.requires_grad_(True)
                    a_t = a_t.requires_grad_(True)
                r = self.reward_net(s_r, a_t).squeeze(0)
                rs.append(r)

                if with_grad:
                    grads = torch.autograd.grad(
                        outputs=r,
                        inputs=(s_r, a_t),
                        grad_outputs=torch.ones_like(r),
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=False,
                    )
                    inv_std = torch.tensor(
                        1.0
                        / np.maximum(
                            self.reward_stat.obs_std, self.reward_stat.std_floor
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                    r_s = grads[0].squeeze(0) * inv_std
                    r_a = grads[1].squeeze(0)
                    r_grads.append((r_s, r_a))
                else:
                    r_grads.append(None)

            # critic value (all states)
            s_c = self.critic_processor(s[: self.config.critic_d_s]).unsqueeze(0)
            if with_grad:
                s_c = s_c.requires_grad_(True)
            v = self.critic(s_c).squeeze(0)
            v_denorm = self.q_stats.Q_std * v + self.q_stats.Q_mean
            vs.append(v_denorm)

            if with_grad:
                grads = torch.autograd.grad(
                    outputs=v,
                    inputs=(s_c,),
                    grad_outputs=torch.ones_like(v),
                    create_graph=False,
                    retain_graph=False,
                )
                inv_std = torch.tensor(
                    1.0
                    / np.maximum(
                        self.critic_stat.obs_std, self.critic_stat.std_floor
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                v_s = grads[0].squeeze(0) * inv_std
                v_grads.append(v_s)
            else:
                v_grads.append(None)

        # -------------------- λ-weighted multi-step returns --------------------
        plan_return = torch.tensor(0.0, device=device)
        coeff_r = [torch.tensor(0.0, device=device) for _ in range(n - 1)]
        coeff_v = [torch.tensor(0.0, device=device) for _ in range(n)]

        # w_H = (1 - gae_lam) * gae_lam^{H-1}
        w = (1.0 - self.config.gae_lam) * self.config.gae_lam          # value for H = 2
        weight_sum = 0.0

        for h in range(2, n + 1):              # H = 2 … n
            # partial = Σ_{t=0}^{h-2} γ^t r[t] + γ^{h-1} V[h-1]
            partial = torch.tensor(0.0, device=device)
            for t in range(h - 1):
                partial = partial + (gamma ** t) * rs[t]
                coeff_r[t] = coeff_r[t] + w * (gamma ** t)

            partial = partial + (gamma ** (h - 1)) * vs[h - 1]
            coeff_v[h - 1] = coeff_v[h - 1] + w * (gamma ** (h - 1))

            plan_return = plan_return + w * partial
            weight_sum += w
            w *= self.config.gae_lam

        Z = max(weight_sum, 1e-8)
        plan_return = plan_return / Z
        for t in range(n - 1):
            coeff_r[t] = coeff_r[t] / Z
        for j in range(n):
            coeff_v[j] = coeff_v[j] / Z

        # -------------------- constraint terms (original meaning of lam) --------------------
        total_c = torch.tensor(0.0, device=device)
        c_grads = []

        for i in range(n - 1):
            s = x[i, : self.config.d_s]
            a = x[i, self.config.d_s :].unsqueeze(0)
            s_next = x[i + 1, : self.config.d_s]

            s_k = self.kernel_processor(s).unsqueeze(0)
            s_next_k = self.kernel_processor(s_next).unsqueeze(0)
            if with_grad:
                s_k = s_k.requires_grad_(True)
                a = a.requires_grad_(True)
                s_next_k = s_next_k.requires_grad_(True)

            c = self.sigmoid(s_k, a, s_next_k).squeeze(0)
            total_c = total_c + c

            if with_grad:
                grads = torch.autograd.grad(
                    outputs=c,
                    inputs=(s_k, a, s_next_k),
                    grad_outputs=torch.ones_like(c),
                    create_graph=True,
                    retain_graph=True,
                )
                inv_std = torch.tensor(
                    1.0
                    / np.maximum(
                        self.kernel_stat.obs_std, self.kernel_stat.std_floor
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                c_s = grads[0].squeeze(0) * inv_std
                c_a = grads[1].squeeze(0)
                c_s_next = grads[2].squeeze(0) * inv_std
                c_grads.append((c_s, c_a, c_s_next))
            else:
                c_grads.append(None)

        total_reward = plan_return - lam * total_c + lam * self.config.delta

        if not with_grad:
            return total_reward, None

        # -------------------- assemble full trajectory gradient --------------------
        gradient = torch.zeros(H, D, device=device)

        # from the r_t terms
        for t in range(n - 1):
            if coeff_r[t] != 0:
                r_s, r_a = r_grads[t]
                g_s, g_a = self.makeGrad(H, r_s, r_a, t)
                gradient = gradient + coeff_r[t] * (g_s + g_a)

        # from the V(s_j) terms
        for j in range(n):
            if coeff_v[j] != 0:
                v_s = v_grads[j]
                g_v = self.makeGrad_Critic(H, v_s, j)
                gradient = gradient + coeff_v[j] * g_v

        # from the constraint terms
        for i in range(n - 1):
            c_s, c_a, c_s_next = c_grads[i]
            g_s, g_a, g_s_next = self.makeGrad(H, c_s, c_a, i, c_s_next)
            gradient = gradient - lam * (g_s + g_a + g_s_next)

        return total_reward, gradient

    # ---------------------------------------------------------------------- public API
    def predict(self, x: torch.Tensor, lam: float):
        
        total_reward, _ = self._compute_gae_style_return(
            x, lam,  with_grad=False
        )
        return total_reward

    def forward(self, x: torch.Tensor, lam: float):
        
        total_reward, gradient = self._compute_gae_style_return(
            x, lam, with_grad=True
        )
        return total_reward, gradient



