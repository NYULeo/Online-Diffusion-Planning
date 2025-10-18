import math
import numpy as np
import torch
from dataclasses import dataclass
from Planners.Backbone.UNet import TemporalUnet
from Planners.Backbone.Sampler import sample_reverse_sde
import pickle
from typing import Optional
from Dataset import get_env
from utils import set_seed
from Planners.Backbone.utils import get_pretrained_planner
from Critic.train_critic import Critic, Critic_Processor, get_CriticName
from Dataset import Planner_Processor
import gymnasium as gym
import os
from Planners.Backbone.Dit import DiT1d
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv 
import mediapy as media


"""
def get_pretrained_planner(planner_name, checkpoint_steps):
      checkpoint_path = f"./Checkpoints/{planner_name}_{checkpoint_steps}.pt"
      if not os.path.exists(checkpoint_path):
          raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      return checkpoint['ema']
"""

def save_trajs(trajs, env_name, specific_env):
    os.makedirs(f'./Rollouts/{env_name}/{specific_env}/', exist_ok=True)
    save_path = f'./Rollouts/{env_name}/{specific_env}/Generated_trajs_Info.pkl'
    with open(save_path, 'wb') as f:
         pickle.dump(trajs, f)
    print(f"trajectories saved")

class ActionSelector:
     def __init__(self, dataset_name, specific_dataset, device):
         self.dataset_name = dataset_name
         self.specific_dataset = specific_dataset
         env, self.d_s, self.d_a = get_env(self.dataset_name, self.specific_dataset)
         self.critic = Critic(self.d_s, self.d_a)
         critic_name = get_CriticName(self.dataset_name, self.specific_dataset)
         critic_state_dict = torch.load(critic_name, map_location = 'cpu')
         self.critic.load_state_dict(critic_state_dict)
         self.critic = self.critic.to(device)  # Move critic to correct device
         self.critic.eval()
         self.critic_processor = Critic_Processor(self.dataset_name, self.specific_dataset)
         self.device = device

     def action_selection(self, current_state, actions):
        q_values = []
        for i in range(len(actions)):
           current_state_norm, act_norm = self.critic_processor.preprocess(current_state, actions[i])
           current_state_norm = torch.tensor(current_state_norm, dtype = torch.float32).unsqueeze(0).to(self.device)
           act_norm = torch.tensor(act_norm, dtype = torch.float32).unsqueeze(0).to(self.device)
           q_value = self.critic(current_state_norm, act_norm)
           q_values.append(q_value.item())
        idx = np.argmax(q_values)
        return actions[idx]


def rollout(env_name, specific_env, horizon, steps_T, eta, episode_length, critic, checkpoint_steps, render = False):
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, critic: {critic}, Checpoint_steps; {checkpoint_steps}")
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     device = "cuda" if torch.cuda.is_available() else "cpu"
     print(f"Using device {device}")
     if critic:
            action_selector = ActionSelector(env_name, specific_env, device)
     else:
            action_selector = None
     
     #get environment
     if(render):
         env, d_s, d_a = get_env(env_name, specific_env, 'rgb_array')
     else:
         env, d_s, d_a = get_env(env_name, specific_env, None)

     #get Planner
     state_dict = get_pretrained_planner(env_name, specific_env, checkpoint_steps)
     if( env_name == 'kitchen'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif (env_name == 'pointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     else:
          raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()

    #get Processor
     planner_processor = Planner_Processor(env_name, specific_env)

     
     #reset
     s0 = env.reset(seed=1)
     s0 = s0[0]['observation']
     current_state = s0
     frames = []
     observations = []
     actions = []
     rewards = []
     for i in range(episode_length):
           current_state_norm = planner_processor.preprocess(current_state)
           #current_state_norm = current_state
           #x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta, device)
           if critic:
                candidates = []
                for j in range(10):
                   
                   x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta,  device = device)
                   action = x[0, d_s:(d_s+d_a)].copy()
                   #action = torch.tanh(action)
                   #action = planner_processor.postprocess(action)
                   candidates.append(action)
                action = action_selector.action_selection(current_state, candidates)
           else:
               x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta,  device = device)
               #print(x[0])
               action = x[0, d_s:(d_s+d_a)].copy()
               #print(action)
               #exit()

               #action = torch.tanh(torch.tensor(action))
               #action = planner_processor.postprocess(action)
               #action = x[0, d_s:(d_s+d_a)].copy()
               
           obs, reward, terminated, truncated, info = env.step(action)
           if(render):
                frames.append(env.render())
           
           observations.append(obs['observation'].copy())
           actions.append(action.copy())
           rewards.append(reward)
           current_state = obs['observation'].copy()
           #print(f"Episode {i} reward: {reward}")
           if(terminated or truncated):
                #print(f"Episode {i} terminated or truncated")
                break
     
     env.close()
     traj = {'observations': np.asarray(observations), 'actions': np.asarray(actions), 'rewards': np.asarray(rewards)}
     traj_info = {'sequence': traj, 'env_name': env_name, 'specific_env': specific_env }
     if(render):
          media.write_video("demo.mp4", frames, fps=50)
     with open('Generated_trajectory.pkl', 'wb') as f:
                pickle.dump(traj_info, f)
     

def rollout_parallel(env_name, specific_env, horizon = 32, steps_T = 500, eta = 0.8, episode_length = 4000, critic = False, checkpoint_steps = 1000000, num_envs=8):
     """
     Run rollout on multiple environments in parallel and save the best trajectory
     
     Args:
         num_envs: Number of parallel environments (default: 4)
     """
     print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, critic: {critic}, Checkpoint_steps: {checkpoint_steps}")
     print(f"Running {num_envs} environments in parallel")
     
     device = "cuda" if torch.cuda.is_available() else "cpu"
     print(f"Using device {device}")
     
     if critic:
         action_selector = ActionSelector(env_name, specific_env, device)
     else:
         action_selector = None
     
     # Create environment factory function
     def make_env():
         env, _, _ = get_env(env_name, specific_env)
         return env
     
     # Create vectorized environment
     vec_env = AsyncVectorEnv([make_env for _ in range(num_envs)])
     
     # Get dimensions from single env
     _, d_s, d_a = get_env(env_name, specific_env)
     
     # Get Planner
     state_dict = get_pretrained_planner(env_name, specific_env, checkpoint_steps)
     if env_name == 'kitchen':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     elif env_name == 'pointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     else:
         raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()
     
     # Get Processor
     planner_processor = Planner_Processor(env_name, specific_env)
     
     # Reset all environments
     s0_vec = vec_env.reset(seed=1)
     current_states = s0_vec[0]['observation']  # Shape: (num_envs, d_s)
     
     # Store trajectories for each environment
     all_rewards = [0.0 for _ in range(num_envs)]
     done_envs = [False for _ in range(num_envs)]
     observations = [[] for _ in range(num_envs)]
     acts = [[] for _ in range(num_envs)]
     rewards = [[] for _ in range(num_envs)]
     
     for i in range(episode_length):
         actions = np.zeros((num_envs, d_a))
         
         # Generate actions for each environment
         for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
                 
             current_state = current_states[env_idx]
             current_state_norm = planner_processor.preprocess(current_state)
             
             if critic:
                 action_candidates = []
                 for j in range(10):
                     x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta, device=device)
                     action = x[0, d_s:(d_s+d_a)].copy()
                     action_candidates.append(action)
                 action = action_selector.action_selection(current_state, action_candidates)
             else:
                 x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta, device=device)
                 action = x[0, d_s:(d_s+d_a)].copy()
             
             actions[env_idx] = action
         
         # Step all environments at once
         obs_vec, rewards_vec, terminated_vec, truncated_vec, info_vec = vec_env.step(actions)
         
         # Update trajectories
         for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
             
             observations[env_idx].append(obs_vec['observation'][env_idx].copy())
             acts[env_idx].append(actions[env_idx].copy())
             rewards[env_idx].append(rewards_vec[env_idx])
             all_rewards[env_idx] += rewards_vec[env_idx]
             
             current_states[env_idx] = obs_vec['observation'][env_idx].copy()
             
             if terminated_vec[env_idx] or truncated_vec[env_idx]:
                 done_envs[env_idx] = True
                 print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
         # Check if all environments are done
         if all(done_envs):
             print("All environments completed!")
             break
         
         if i % 50 == 0:
             active_count = sum(not d for d in done_envs)
             if active_count > 0:
                 print(f"Step {i}: Active envs: {active_count}")
     
     vec_env.close()
     
     # Find the trajectory with the maximum reward
     trajs = [[] for _ in range(num_envs)]
     for env_idx in range(num_envs):
         trajs[env_idx] = {
             'observations': np.asarray(observations[env_idx].copy()),
             'actions': np.asarray(acts[env_idx].copy()),
             'rewards': np.asarray(rewards[env_idx].copy())
         }
     best_idx = np.argmax(all_rewards)
     best_reward = all_rewards[best_idx]
     best_trajectory = trajs[best_idx]
     
     print(f"\n{'='*60}")
     print(f"Results from {num_envs} parallel rollouts:")
     print(f"{'='*60}")
     for env_idx in range(num_envs):
         print(f"  Env {env_idx}: Total reward = {all_rewards[env_idx]:.4f}, Steps = {len(trajs[env_idx])}")
     print(f"{'='*60}")
     print(f"Best trajectory: Env {best_idx} with reward = {best_reward:.4f}")
     print(f"Average reward: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
     print(f"{'='*60}\n")
     
     # Save the best trajectory in the same format as single rollout
     trajs_info = {
         'best_traj': best_trajectory,
         'trajs': trajs,
         'env_name': env_name,
         'specific_env': specific_env,
         'total_reward': best_reward,
         'num_envs_tested': num_envs,
         'all_rewards': all_rewards
     }
     save_trajs(trajs_info, env_name, specific_env)
    
     





# ---- 4) Example usage (fill ScoreWrapper first) ----
if __name__ == "__main__":
    set_seed(1)
    horizon = 32
    env_name = 'kitchen'
    specific_train_dataset = 'partial'
    #rollout(env_name, specific_train_dataset, horizon, steps_T = 500, eta = 0.8, episode_length  = 2000, critic = False, checkpoint_steps = 1000000, render = True)
    rollout_parallel(env_name, specific_train_dataset, horizon, steps_T = 500, eta = 0.8, episode_length  = 4000, critic = False, checkpoint_steps = 990000, num_envs = 8)
  
