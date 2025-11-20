
import numpy as np
from utils import set_seed
import mediapy as media
from Dataset import get_env
import pickle
import os
from gymnasium.vector import AsyncVectorEnv
from Dataset import get_dataset

def render(dataset_name, specific_dataset, traj):
     env, _, _ = get_env(dataset_name, specific_dataset, render_mode = 'rgb_array')
     env.reset(seed=1)
     frames = []
     for i in range(len(traj['actions'])):
          action = traj['actions'][i]
            #action = np.clip(action, -1.0, 1.0)
          _, _, terminated, truncated, _ = env.step(action)
          frames.append(env.render())
          if terminated or truncated:
               break
     media.write_video("demo2.mp4", frames, fps=50)
     env.close()





def spare_reward_kitchen(rewards):
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
     """
     with open('Rollouts/pointmaze/medium/Generated_trajs_Info.pkl', 'rb') as f:
        info = pickle.load(f)
     trajs = info['trajs']
     best_traj = info['best_traj']
     """
     data = get_dataset('pointmaze', 'medium')
     trajs = data.get_trajectories()
     Goals = []
     seen = set()
     for traj in trajs:
        rewards = traj['rewards']
        for obs, rew in zip(traj['observations'], rewards):
           if rew == 1:
               goal = tuple(np.round(obs[:2], 3))   # round to avoid float jitter
               if goal not in seen:
                   seen.add(goal)
                   Goals.append(goal)
     Goals = np.array(Goals)
     MIN_DIST = 0.1  # tweak threshold
     filtered = []
     for goal in Goals:
         if all(np.linalg.norm(goal - kept) >= MIN_DIST for kept in filtered):
             filtered.append(goal)
     Goals = np.array(filtered)
     print(len(Goals))
     #render('pointmaze', 'medium', trajs[0])
     """
     env_name = info['env_name']
     specific_env = info['specific_env']
     all_rewards = []
     num_envs = info['num_envs_tested']
     for traj in trajs:
           rewards = traj['rewards']
           new_rewards = spare_reward_kitchen(rewards)
           print(reward_checker(rewards, new_rewards))
           traj['rewards'] = new_rewards
           all_rewards.append(np.sum(new_rewards))
     
     best_idx = np.argmax(all_rewards)
     best_reward = all_rewards[best_idx]
     best_trajectory = trajs[best_idx]
     trajs_info = {
         'best_traj': best_trajectory,
         'trajs': trajs,
         'env_name': env_name,
         'specific_env': specific_env,
         'total_reward': best_reward,
         'num_envs_tested': num_envs,
         'all_rewards': all_rewards
     }
     render(env_name, specific_env, best_trajectory)
     print(best_reward)
     
     save_path = f'./Rollouts/kitchen/partial/Generated_trajs_Info.pkl'
     with open(save_path, 'wb') as f:
         pickle.dump(trajs_info, f)
     """


    
     
