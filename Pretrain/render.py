
import numpy as np
from utils import set_seed
import mediapy as media
from Dataset import get_env
import pickle
import os
from gymnasium.vector import AsyncVectorEnv
from Dataset import get_dataset

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
     env.reset()
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
     print(len(frames))
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



"""
if __name__ == "__main__":
     set_seed(1)
     data = get_dataset('kitchen', 'complete')
     trajs = data.get_trajectories()
     render('kitchen', 'partial', trajs[0])
"""





import ogbench
import numpy as np
import imageio

def find_success_episode(env, data, max_search=1000):
    """
    Find a successful trajectory in the dataset when replayed
    in the goal-conditioned environment.
    Returns a dict with keys 'actions' and 'observations' if found, else None.
    """
    actions = data["actions"]
    obs = data["observations"]
    terminals = data["terminals"]

    start = 0
    count = 0

    # Episodes are separated by terminals = 1
    for i, t in enumerate(terminals):
        if t == 1:
            segment_actions = actions[start:i+1]
            segment_obs = obs[start:i+1]

            # Reset env for each candidate
            env.reset()

            success = False
            for a in segment_actions:
                _, _, terminated, truncated, info = env.step(a)
                if info.get("success", 0) == 1:
                    success = True
                    break
                if terminated or truncated:
                    break

            if success:
                return {
                    "actions": segment_actions,
                    "observations": segment_obs
                }

            start = i + 1
            count += 1
            if count >= max_search:
                break

    return None

def render_success_episode(dataset_name, task_id=1, out_video="demo.mp4"):
    """
    Find one successful episode and render it to a video file.
    Args:
        dataset_name (str): e.g. "cube-single-play-v0"
        task_id (int): which goal to evaluate (1..5)
        out_video (str): output video filename
    """
    # Load the goal-conditioned environment + dataset
    env, train_data, _ = ogbench.make_env_and_datasets(dataset_name, render_mode="rgb_array")
    
    # Reset once with a goal
    env.reset(options=dict(task_id=task_id))
    
    print(f"Searching for a success episode in {dataset_name} (task {task_id}) ...")
    success_traj = find_success_episode(env, train_data)
    
    if success_traj is None:
        print("No successful trajectory found.")
        env.close()
        return
    
    print("Found a successful trajectory! Rendering ...")

    # Reset env with the same goal again
    env.reset(options=dict(task_id=task_id))

    frames = []
    for action in success_traj["actions"]:
        _, reward, terminated, truncated, info = env.step(action)
        frame = env.render()  # returns rgb_array
        frames.append(frame)
        if info.get("success", 0) == 1 or terminated or truncated:
            break

    # Save to video
    imageio.mimsave(out_video, frames, fps=30)
    print(f"Saved video to {out_video} (steps={len(frames)})")

    env.close()


if __name__ == "__main__":
    # Example usage
    render_success_episode("cube-single-play-singletask-v0", task_id=2, out_video="cube_success.mp4")




     
"""
     with open('Rollouts/pointmaze/medium/Generated_trajs_Info.pkl', 'rb') as f:
        info = pickle.load(f)
     trajs = info['trajs']
     best_traj = info['best_traj']
"""
     
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


    
     
