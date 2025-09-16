from utils import set_seed
import mediapy as media
import warnings
import pickle
warnings.filterwarnings("ignore", category=UserWarning)
from Dataset import get_env, get_dataset
from utils import SAStats
import numpy as np
import matplotlib.pyplot as plt
#load the trajectory

"""

with open('Generated_trajectory.pkl', 'rb') as f:
       traj_info = pickle.load(f)


#get action sequence
sequence = traj_info['sequence']
Gen_actions = []
for i in range(len(sequence)):
      Gen_actions.append(sequence[i]['action'])
"""




data = get_dataset('kitchen', 'partial')
traj = data.get_trajectories()
actions = traj[4]['actions']

#get environment
env, d_s, d_a= get_env('kitchen', 'partial')



#start animation
set_seed(0)
env.reset()
frames = []
for i in range(len(actions)):
    #action = np.random.uniform(-1.0, 1.0, d_a)
    #action = np.clip(actions[i], -1.0, 1.0)
    obs, rew, terminated, truncated, info = env.step(actions[i])
    frames.append(env.render())
    if terminated or truncated:
        break

media.write_video("demo.mp4", frames, fps=30)


"""
data = get_dataset('kitchen', 'partial')
traj = data.get_trajectories()
actions = traj[0]['actions']
actions = np.array(actions)
Gen_actions = np.array(Gen_actions)




plt.hist(actions.flatten(), bins=80, alpha=0.5, label="dataset actions")
plt.hist(Gen_actions.flatten(), bins=80, alpha=0.5, label="rollout actions")
plt.legend()
plt.show()
"""

