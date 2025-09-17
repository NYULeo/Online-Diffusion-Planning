import math
import numpy as np
import torch
from dataclasses import dataclass
from Backbone.UNet import TemporalUnet
from Backbone.Sampler import sample_reverse_sde
import pickle
from typing import Optional
from Dataset import get_env
from utils import set_seed
from Backbone.utils import get_pretrained_planner
from train_critic import Critic, Critic_Processor, get_CriticName
from Dataset import Planner_Processor, get_PlannerName
import gymnasium as gym
import os
from Backbone.Dit import DiT1d
import mediapy as media


"""
def get_pretrained_planner(planner_name, checkpoint_steps):
      checkpoint_path = f"./Checkpoints/{planner_name}_{checkpoint_steps}.pt"
      if not os.path.exists(checkpoint_path):
          raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      return checkpoint['ema']
"""


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


def rollout(env_name, specific_env, horizon, steps_T, eta, episode_length, critic, checkpoint_steps):
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
     env, d_s, d_a = get_env(env_name, specific_env)

     #get Planner
     planner_name = get_PlannerName(env_name, specific_env)
     state_dict = get_pretrained_planner(planner_name, checkpoint_steps)
     if( env_name == 'kitchen'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)    
     else:
          raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()

    #get Processor
     planner_processor = Planner_Processor(env_name, specific_env)

     
     #reset
     s0 = env.reset()
     s0 = s0[0]['observation']
     current_state = s0
     play_seq = []
     frames = []
     for i in range(episode_length):
           current_state_norm = planner_processor.preprocess(current_state)
           #current_state_norm = current_state
           #x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta, device)
           if critic:
                actions = []
                for j in range(10):
                   x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta,  device = device)
                   action = x[0, d_s:(d_s+d_a)].copy()
                   #action = torch.tanh(action)
                   #action = planner_processor.postprocess(action)
                   actions.append(action)
                action = action_selector.action_selection(current_state, actions)
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
           frames.append(env.render())
           step = {'observation': obs['observation'].copy(), 'action':action.copy(), 'reward': reward}
           play_seq.append(step)
           current_state = obs['observation'].copy()
           #print(f"Episode {i} reward: {reward}")
           if(terminated or truncated):
                #print(f"Episode {i} terminated or truncated")
                break
     
     traj_info = {'sequence': play_seq, 'env_name': env_name, 'specific_env': specific_env }
     media.write_video("demo.mp4", frames, fps=50)
     with open('Generated_trajectory.pkl', 'wb') as f:
                pickle.dump(traj_info, f)
     
           


# ---- 4) Example usage (fill ScoreWrapper first) ----
if __name__ == "__main__":
    set_seed(1)
    horizon = 32
    env_name = 'kitchen'
    specific_train_dataset = 'partial'

    rollout(env_name, specific_train_dataset, horizon, steps_T = 500, eta = 0.8, episode_length  = 500, critic = False, checkpoint_steps = 990000)


