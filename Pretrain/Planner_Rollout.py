import math
import numpy as np
import torch
from dataclasses import dataclass
from Backbone import UNet1D, sample_reverse_sde, sample_pf_ode
import pickle
from typing import Optional
from Dataset import get_env
from utils import set_seed
from train_critic import Critic, Critic_Processor, get_CriticName
from pretrain_planner import Planner_Processor, get_PlannerName
import gymnasium as gym


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
           current_state_norm, act_norm= self.critic_processor.preprocess(current_state, actions[i])
           current_state_norm = torch.tensor(current_state_norm, dtype = torch.float32).unsqueeze(0).to(self.device)
           act_norm = torch.tensor(act_norm, dtype = torch.float32).unsqueeze(0).to(self.device)
           q_value = self.critic(current_state_norm, act_norm)
           q_values.append(q_value.item())
        idx = np.argmax(q_values)
        return actions[idx]


def rollout(env_name, specific_env, horizon, steps_T, eta, episode_length : int = 1000):
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     print(f"Horizon: {horizon}, step_T: {steps_T}")
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     device = "cuda" if torch.cuda.is_available() else "cpu"
     print(f"Using device {device}")
     #action_selector = ActionSelector(env_name, specific_env, device)
     planner_processor = Planner_Processor(env_name, specific_env)

     #get environment
     env, d_s, d_a = get_env(env_name, specific_env)

     #get Planner 
     planner_name = get_PlannerName(env_name, specific_env)
     model = UNet1D(input_dim=(d_s+d_a)*horizon).to(device)
     state_dict = torch.load(planner_name, map_location='cpu')
     model.load_state_dict(state_dict)
     model.eval()

     
     #reset
     s0 = env.reset()
     s0 = s0[0]['observation']
     current_state = s0
     play_seq = []
     
     for i in range(episode_length):
           actions = []
           current_state_norm = planner_processor.preprocess(current_state)
           x = sample_pf_ode(current_state_norm, model, d_s, d_a, horizon, steps_T, device)
           """
           for j in range(10):
                x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta,  device = device)
                action = planner_processor.postprocess(x[d_s:(d_s+d_a)].copy())
                actions.append(action)
           """
           #action = action_selector.action_selection(current_state, actions)
           action = planner_processor.postprocess(x[d_s:(d_s+d_a)].copy())
           obs, reward, terminated, truncated, info = env.step(action)
           step = {'observation': obs['observation'].copy(), 'action':action.copy(), 'reward': reward}
           play_seq.append(step)
           current_state = obs['observation'].copy()
           #print(f"Episode {i} reward: {reward}")
           if(terminated or truncated):
                #print(f"Episode {i} terminated or truncated")
                break
     
     traj_info = {'sequence': play_seq, 'env_name': env_name, 'specific_env': specific_env }

     with open('Generated_trajectory.pkl', 'wb') as f:
                pickle.dump(traj_info, f)
     
           


# ---- 4) Example usage (fill ScoreWrapper first) ----
if __name__ == "__main__":
    set_seed(1)
    horizon = 32
    env_name = 'kitchen'
    specific_env = 'partial'

    rollout(env_name, specific_env, horizon, steps_T = 100, eta = 0.3, episode_length  = 500)

