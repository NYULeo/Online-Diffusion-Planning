
import numpy as np
from utils import set_seed
import mediapy as media
from Dataset import get_env
import pickle


def render(dataset_name, specific_dataset, traj):
     env, _, _ = get_env(dataset_name, specific_dataset, render_mode = 'rgb_array')
     env.reset()
     frames = []
     for i in range(len(traj['actions'])):
         action = traj['actions'][i]
         #action = np.clip(action, -1.0, 1.0)
         _, _, terminated, truncated, _ = env.step(action)
         frames.append(env.render())
         if terminated or truncated:
               break
     media.write_video("demo.mp4", frames, fps=30)
     env.close()





if __name__ == "__main__":
     with open('Generated_trajectory.pkl', 'rb') as f:
        info = pickle.load(f)
     #trajs = info['trajs']
     #traj = trajs[5]
     traj = info['sequence']
     obs = []
     acts = []
     rewards = []
     for step in traj:
            obs.append(step['observation'].copy())
            acts.append(step['action'].copy())
            rewards.append(step['reward'])
     obs = np.array(obs)
     acts = np.array(acts)
     rewards = np.array(rewards)
     traj = {'observations': obs, 'actions': acts, 'rewards': rewards}
     set_seed(0)
     render(dataset_name = 'pointmaze', specific_dataset = 'medium', traj = traj)
     
