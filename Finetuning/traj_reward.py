import sys
import os

from torch.optim.optimizer import required
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import torch
import torch.nn as nn
from Pretrain.Rewards.nets import SimpleReward
from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel
from Pretrain.Critic.nets import Critic
from Finetuning.utils import get_reward_model, get_kernel, get_reward_stats, get_kernel_stats, get_critic_model, get_critic_stats
from typing import Optional
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np



@dataclass
class RewardConfig:
    """Configuration for the adjoint matching fine‑tuner."""
    beta: float
    min_log_prob: float
    explore: bool = True
    gamma: float = 0.8
    device = None
    d_s: int = 0 
    d_a: int = 0
    num_hidden_layers_kernel: int = 2
    critic_d_s: int = 0
    delta: Optional[float] = None 
    

class TotalReward(nn.Module):
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int):
        super().__init__()
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
        self.config.device = device
        self.reward_net = SimpleReward(obs_dim, act_dim).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()
        self.kernels = []
        self.config.delta = F.softplus(torch.tensor(0.0, requires_grad = False), beta = self.config.beta).to(self.config.device)


        
        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        for i in range(len(kernel_state_dicts)):
                kernel_net = RobustTransitionKernel(obs_dim, act_dim, self.config.num_hidden_layers_kernel).to(self.config.device)
                kernel_net.load_state_dict(kernel_state_dicts[i])
                kernel_net.eval()
                self.kernels.append(kernel_net)
        
        
        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)
       

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        if(not self.config.explore):
              self.config.gamma = 0.0
        
    def get_beta(self):
        return self.config.beta

    def sigmoid(self, s, a, s_next):
        total = torch.tensor([0.0], device = self.config.device, requires_grad = True)
        for i in range(len(self.kernels)):
            mu, log_std = self.kernels[i](s, a)
            lp = self.kernels[i].log_prob(s_next, mu, log_std)
            #lp = self.kernels[i].prob(s_next, mu, log_std)
            total = total + lp 
        avg = total / len(self.kernels)
        x =  self.config.min_log_prob - avg
        c = F.softplus(x, beta = self.config.beta)
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
            total_reward += (1/H)*(r.squeeze(0)) - lam  * ( (1/(H-1)) * c.squeeze(0))
        
        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(False)
        a = x[H-1][self.config.d_s:].unsqueeze(0).requires_grad_(False)
        r = self.reward_net(s_norm_reward, a)
        total_reward +=  (1/H) * (r.squeeze(0))
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
            
            gradient +=  (1/H)*((r_s_grad + r_a_grad)) - lam * (1/(H-1)) * (c_s_grad + c_a_grad + c_s_next_grad)
            
            total_reward += (1/H)*(r.squeeze(0)) - lam  * ( (1/(H-1)) * c.squeeze(0))
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

       
       
        gradient += (1/H) * ((r_s_grad + r_a_grad)) 
        total_reward +=  (1/H) * (r.squeeze(0))
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient


class TotalReward_Critic(nn.Module):
    def __init__(self, device, config: RewardConfig, dataset_name: str, specific_dataset: str, reward_checkpoint: int, kernel_checkpoint: int, critic_checkpoint: int):
        super().__init__()
        self.config = config
        reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
        self.config.device = device
        self.reward_net = SimpleReward(obs_dim, act_dim).to(self.config.device)
        self.reward_net.load_state_dict(reward_state_dict)
        self.reward_net.eval()
        self.kernels = []
        self.config.delta = F.softplus(torch.tensor(0.0, requires_grad = False), beta = self.config.beta).to(self.config.device)


        critic_state_dict, critic_obs_dim = get_critic_model(dataset_name, specific_dataset, critic_checkpoint)
        self.critic = Critic(critic_obs_dim).to(self.config.device)
        self.critic.load_state_dict(critic_state_dict)
        self.critic.eval()
        print(critic_state_dict)
        exit()

        kernel_state_dicts, obs_dim, act_dim = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
        for i in range(len(kernel_state_dicts)):
                kernel_net = RobustTransitionKernel(obs_dim, act_dim, self.config.num_hidden_layers_kernel).to(self.config.device)
                kernel_net.load_state_dict(kernel_state_dicts[i])
                kernel_net.eval()
                self.kernels.append(kernel_net)
        
        self.reward_stat = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
        self.kernel_stat = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)
        self.critic_stat = get_critic_stats(dataset_name, specific_dataset, critic_checkpoint)
       

        self.config.d_s = obs_dim
        self.config.d_a = act_dim
        self.config.critic_d_s = critic_obs_dim
        if(not self.config.explore):
              self.config.gamma = 0.0
        
    def get_beta(self):
        return self.config.beta

    def sigmoid(self, s, a, s_next):
        total = torch.tensor([0.0], device = self.config.device, requires_grad = True)
        for i in range(len(self.kernels)):
            mu, log_std = self.kernels[i](s, a)
            #lp = self.kernels[i].log_prob(s_next, mu, log_std)
            lp = self.kernels[i].log_prob(s_next, mu, log_std)
            total = total + lp 
        avg = total / len(self.kernels)
        x =  self.config.min_log_prob - avg
        c = F.softplus(x, beta = self.config.beta)
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
            total_reward += (1/H)*(r.squeeze(0)) - lam  * ( (1/(H-1)) * c.squeeze(0))
        
        s = x[H-1][:self.config.d_s]
        s_norm_reward = self.reward_processor(s).unsqueeze(0).requires_grad_(False)
        a = x[H-1][self.config.d_s:].unsqueeze(0).requires_grad_(False)
        r = self.reward_net(s_norm_reward, a)
        final_s_critic = x[H-1][:self.config.critic_d_s]
        final_s_norm_critic = self.critic_processor(final_s_critic).unsqueeze(0).requires_grad_(False)
        v = self.critic(final_s_norm_critic)
        total_reward +=  (1/H) * (r.squeeze(0)) + v.squeeze(0)
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
            
            gradient +=  (1/H)*((r_s_grad + r_a_grad)) - lam * (1/(H-1)) * (c_s_grad + c_a_grad + c_s_next_grad)
            
            total_reward += (1/H)*(r.squeeze(0)) - lam  * ( (1/(H-1)) * c.squeeze(0))
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
        
       
        gradient += (1/H) * ((r_s_grad + r_a_grad))  + grad_critic
        total_reward +=  (1/H) * (r.squeeze(0)) + v.squeeze(0)
        total_reward = total_reward + (lam  * self.config.delta)
        return total_reward, gradient




"""
import pickle

save_path = f'./Pretrain/Rollouts/{'pointmaze'}/{'medium'}/Generated_trajs_Info.pkl'
with open(save_path, 'rb') as f:
    data = pickle.load(f)
gen_trajs = data['trajs']






reward_model_state_dict, obs_dim, act_dim, name = get_pretrained_reward('pointmaze', 44000, 'medium')
reward_model = Reward(obs_dim, act_dim)
reward_model.load_state_dict(reward_model_state_dict)
reward_model.eval()
stats = get_pretrained_reward_stats(name)

total = 0.0
for i in range(len(gen_trajs)):
     traj = gen_trajs[i]
     traj_reward = 0.0
     Grad_sum = 0.0
     for j in range(len(traj['actions'])):
          obs = traj['observations'][j].copy()
          action = traj['actions'][j].copy()
          obs_norm = stats.norm_obs(obs)
          action_norm = action
          obs_norm = torch.tensor(obs_norm, dtype = torch.float32, requires_grad = True).unsqueeze(0)
          action_norm = torch.tensor(action_norm, dtype = torch.float32, requires_grad = True).unsqueeze(0)
          pred =   reward_model(obs_norm, action_norm)
          grad = torch.autograd.grad(
                 outputs=pred,
                 inputs=(obs_norm, action_norm),
                 grad_outputs=torch.ones_like(pred),
                 create_graph=False,
                 retain_graph=False)
          grad_obs = grad[0].squeeze(0)
          grad_action = grad[1].squeeze(0)
          Grad_sum += grad_obs.norm().item() + grad_action.norm().item()
          traj_reward += pred.item()
     print(f"Grad_sum: {Grad_sum / len(traj['actions'])}")
     traj_reward = traj_reward / len(traj['actions'])
     print(f"Traj {i} reward: {traj_reward}")
     total += traj_reward
     
total = total / len(gen_trajs)
print(f"Complete Total reward: {total}")

"""