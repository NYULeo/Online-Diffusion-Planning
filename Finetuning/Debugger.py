

"""
    @torch.no_grad()
    def sample_Trajs(self, s0: torch.Tensor) -> Tuple[List[List[torch.Tensor]], float]:
         s0 = s0.to(self.device)
         L = len(s0)
         Trajs = []
         with self.accelerator.split_between_processes(s0) as local_s0:
            local_results = []
            for s0_single in local_s0:
                  s0_single = s0_single.to(self.device)
                  traj = self.sample_Traj(s0_single)  # one trajectory per state
                  local_results.append(traj)
            Trajs.extend(self.accelerator.gather_for_metrics(local_results))
         
         Total_C = 0.0
         for i in range(len(Trajs)):
            x = Trajs[i][len(Trajs[i])-1].squeeze(0)
            x = x.to(self.device)
            c = self.get_C(x)
            Total_C += c
         avg_C = Total_C / L
         self.Lam.update(avg_C)
         return Trajs, avg_C


    def step(self, s0: torch.Tensor) -> float:
        Trajs, avg_C = self.sample_Trajs(s0)
        self.optimizer.zero_grad()
        Loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_reward = 0.0
        for i in range(len(s0)):
            adjoints, reward = self.make_a(Trajs[i])
            loss = self.adjoint_matching_loss(Trajs[i], adjoints)
            Loss += loss
            total_reward += reward
        avg_reward = total_reward / len(s0)
        Loss = Loss / len(s0)
        self.accelerator.backward(Loss)
        self.accelerator.clip_grad_norm_(self.new_score_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return Loss.detach().cpu().item(), avg_reward, avg_C
       
    def step(self, s0_batch: Tensor) -> Tuple[float, float, float]:
        # Split batch across processes
        with self.accelerator.split_between_processes(s0_batch) as local_s0:
            local_losses = []
            local_rewards = []
            local_avgCs = []

            for s0 in local_s0:
                traj = self.sample_Traj(s0)
                adjoints, reward = self.make_adjoint(traj)
                loss = self.adjoint_matching_loss(traj, adjoints)
                local_losses.append(loss)
                local_rewards.append(reward)
                # compute C for final state
                final_x = traj[-1].squeeze(0).to(self.device)
                C_val = self.reward_model.get_c(final_x)
                local_avgCs.append(C_val)

        # gather lists from all processes
        all_losses = self.accelerator.gather_for_metrics(local_losses, use_gather_object=True)
        all_rewards = self.accelerator.gather_for_metrics(local_rewards, use_gather_object=True)
        all_avgCs = self.accelerator.gather_for_metrics(local_avgCs, use_gather_object=True)

        if self.accelerator.is_main_process:
            total_loss = float(sum(all_losses) / len(all_losses))
            total_reward = float(sum(all_rewards) / len(all_rewards))
            total_avgC = float(sum(all_avgCs) / len(all_avgCs))
            self.Lam.update(total_avgC)
            return total_loss, total_reward, total_avgC

    


    
    def make_a(self, X):
        X = [x.to(self.device) for x in X]
        steps_T = len(X)
        X_reversed = X[::-1] 
        a = []
        self.reward_model.eval()
        T = X_reversed[0].to(self.device)
        T_squeezed = T.squeeze(0) 
        reward, gradient = self.reward_model(T_squeezed, self.Lam.get_lam())
        gradient_flat = -1 * gradient.view(-1)  # [H*dim]
        a.append(gradient_flat)
        for i in range(steps_T - 1):
            t_now, t_next = self.t_asc[i], self.t_asc[i + 1]
            dt = (t_next - t_now)
            T = X_reversed[i].to(self.device)
            T.requires_grad_(True)
            
            try:
                Jov = self.compute_jacobian_vectorized(T, i)
            except Exception as e:
                print(f"Vectorized Jacobian failed for step {i}, using fallback: {e}")
                Jov = self._compute_jacobian_elementwise(T, i)
            
            current_a = a[i].to(self.device)  # [H*dim]

        
            # Compute: a + dt * (k[i] * a + 2 * k[i] * Jov @ a)
            new_a = current_a + dt * (self.k[i] * current_a + 2 * self.k[i] * (Jov @ current_a))
            a.append(new_a)
            
        a.reverse()
        return a, reward.item()
    """




"""
        all_loss_tensors = self.accelerator.gather(local_loss_tensors)
        all_rewards = self.accelerator.gather_for_metrics(local_rewards, use_gather_object=True)
        #print(f"All loss tensors: {all_loss_tensors}")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
             # Compute average reward for logging
             avg_reward = float(sum(all_rewards) / len(all_rewards))
             all_loss_tensors.to(self.device)
             print(f"All loss tensors: {all_loss_tensors}")
             loss_for_backprop = all_loss_tensors.mean()

    
             #all_loss_tensors = [loss_tensor.to(self.device) for loss_tensor in all_loss_tensors]
             #print(f"All loss tensors: {all_loss_tensors}")
             #loss_for_backprop = torch.stack(all_loss_tensors).mean().to(self.device)
             
        

             self.optimizer.zero_grad()
             print(f"Loss before backward: {loss_for_backprop}")
             self.accelerator.backward(loss_for_backprop)
             self.accelerator.clip_grad_norm_(self.new_score_net.parameters(), max_norm=1.0)
             self.optimizer.step()
             self.scheduler.step()
             print(f"Loss after backward: {loss_for_backprop}")

             # For logging compute float of 
             # loss
             avg_loss = loss_for_backprop.detach().item()
             return avg_loss, avg_reward, total_avgC
    """
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from accelerate import Accelerator
from Pretrain.Rewards.nets import Reward
from Pretrain.Rewards.Reward_Backbone import get_pretrained_reward, get_pretrained_reward_stats
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed before everything else
seed = 42
set_seed(seed)





accelerator = Accelerator()
device = accelerator.device
rank = accelerator.process_index

torch.manual_seed(seed + rank)
torch.cuda.manual_seed_all(seed + rank)
np.random.seed(seed + rank)
random.seed(seed + rank)


reward_state_dict, obs_dim, act_dim, reward_name = get_pretrained_reward('pointmaze', 44000, 'medium')
reward_net = Reward(obs_dim, act_dim)
reward_net.load_state_dict(reward_state_dict)
reward_net.eval()
reward_stats = get_pretrained_reward_stats(reward_name)



s = torch.tensor([ [1, 2, 3, 4], [4,5,6, 7], [7,8,9,10], [10, 11, 12, 13]], dtype = torch.float32)
a = torch.tensor([[1,2], [3,4], [5,6], [7,8]], dtype = torch.float32)

class SimpleDataset(Dataset):
    def __init__(self, s, a):
        self.s = s
        self.a = a
    
    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, idx):
        return self.s[idx], self.a[idx]

dataset = SimpleDataset(s, a)
generator = torch.Generator()
generator.manual_seed(seed)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(len(dataloader))
reward_net, dataloader = accelerator.prepare(reward_net, dataloader)
reward_net.to(device)
for batch in dataloader:
     s, a = batch
     s = s.to(device)
     a = a.to(device)
     batch_data = list(zip(s, a))

     base_reward_net = accelerator.unwrap_model(reward_net)
     with accelerator.split_between_processes(batch_data) as local_batch:
        local_rewards = []
        local_s_list = []
        local_a_list = []
        print(f"Local_batch: {len(local_batch)}")
        for s_item, a_item in local_batch:
            local_s_list.append(s_item)
            local_a_list.append(a_item)
        
        if len(local_s_list) > 0:
            # Stack into tensors
            local_s = torch.stack(local_s_list)
            local_a = torch.stack(local_a_list)
            
            # Run inference
            with torch.no_grad():
                local_reward = base_reward_net(local_s, local_a)
                print(f'Local_reward: {local_reward}')
            local_rewards.append(local_reward)
    
     accelerator.wait_for_everyone()
     all_rewards = accelerator.gather_for_metrics(local_rewards, use_gather_object=True)
     if accelerator.is_main_process:
        # Concatenate all rewards
        all_rewards = [r.to(device) if isinstance(r, torch.Tensor) else r for r in all_rewards]
        all_rewards_tensor = torch.cat(all_rewards, dim=0)
        print(f"Collected rewards: mean={all_rewards_tensor.mean().item():.4f}, shape={all_rewards_tensor.shape}")
     exit()






