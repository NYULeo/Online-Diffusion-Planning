from utils import set_seed
import mediapy as media
import warnings
import pickle
warnings.filterwarnings("ignore", category=UserWarning)
from Dataset import get_env, get_dataset
from utils import SAStats
import numpy as np

#load the trajectory
with open('Generated_trajectory.pkl', 'rb') as f:
    traj_info= pickle.load(f)


#get action sequence
sequence = traj_info['sequence']
actions = []
for i in range(len(sequence)):
      actions.append(sequence[i]['action'])


#get environment
env, d_s, d_a= get_env(traj_info['env_name'], traj_info['specific_env'])




#start animation
set_seed(0)
env.reset()
frames = []
for i in range(len(actions)):
    #action = np.random.uniform(-1.0, 1.0, d_a)
    action = np.clip(action[i], -1.0, 1.0)
    obs, rew, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        break

media.write_video("demo.mp4", frames, fps=30)


