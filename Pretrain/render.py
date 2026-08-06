
import sys
import os

from torch._higher_order_ops.invoke_subgraph import trace_joint_graph_as_bwd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import numpy as np
from Pretrain.utils import set_seed
from Finetuning.utils import get_trajs
import mediapy as media
from Dataset import get_env
import pickle
import os
from gymnasium.vector import AsyncVectorEnv
from Dataset import get_dataset
import gymnasium as gym

def spare_reward_checker(rewards):
     Temp = []
     for i in range(1, len(rewards)):
          if(rewards[i] == rewards[i-1]+1):
               Temp.append(i)
     new_rewards = [0]*len(rewards)
     for i in range(len(rewards)):
          if(i in Temp):
               new_rewards[i] = 1
          else:
               new_rewards[i] = 0
     return np.array(new_rewards, dtype = np.float64) 

def get_normalized_score(rewards):
    total = 0.0
    for i in range(len(rewards)):
        temp = 0.0
        for j in range(len(rewards)):
            #if(trajs[i]['rewards'][j] == 1):
            temp += (0.99**j) * rewards[j]
        total += temp
    avg_discounted_return = total / len(rewards)
    # 5. Compute normalized score
    normalized_score = 100 * avg_discounted_return 
    #print(f"Normalized Score: {normalized_score:.2f}")
    return normalized_score

def render(dataset_name, specific_dataset, traj):
     env, _, _ = get_env(dataset_name, specific_dataset, render_mode = 'rgb_array')
     #env = gym.make("antmaze-medium-v0") 
     obs0 = traj["observations"][0]

     env.reset(seed=0)  # optional fixed seed for determinism

    
     frames = []
     rewards = []
     for i in range(len(traj['actions'])):
          action = traj['actions'][i]
            #action = np.clip(action, -1.0, 1.0)
          _, reward, terminated, truncated, _ = env.step(action)
          rewards.append(reward)
          frames.append(env.render())
          if terminated or truncated:
               break
     print(sum(spare_reward_checker(rewards)))
     #print(rewards)
     #print(len(frames))
     media.write_video("demo2.mp4", frames, fps=50)
     env.close()


def reward_checker(rewards, new_rewards):
         if(len(rewards) != len(new_rewards)):
               return False
         for i in range(1, len(rewards)):
              if(rewards[i] == rewards[i-1]+1):
                  if(new_rewards[i] !=1):
                      return False
              else:
                  if(new_rewards[i] != 0):
                      return False
         return True

def check_speration(trajs):
    print('Checking Separation')
    for i in range(len(trajs)-1):
     states_1 = trajs[i]['observations']
     states_2 = trajs[i+1]['observations']
     if (np.array_equal(states_1[len(states_1)-1], states_2[0])):
         print(i)




if __name__ == "__main__":
     set_seed(1)
     dataset_name = 'ogpointmaze'
     specific_dataset = 'medium'
     task_id = 1
     env, _, _ = get_env(dataset_name, specific_dataset, render_mode = 'rgb_array')
     data = get_dataset(dataset_name, specific_dataset, task_id = task_id, mode = 'critic')
     count = 0
     trajs = data.get_trajectories()
     



