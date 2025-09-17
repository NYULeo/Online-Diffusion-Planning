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




#print(states_1[len(states_1)-1] == states_2[0])
#print(np.array_equal(states_1[len(states_1)-1], states_2[0]))



data = get_dataset('PointMaze', 'medium')
traj = data.get_trajectories()
#actions = traj[311]['actions']
for i in range(len(traj)-1):
     states_1 = traj[i]['observations']
     states_2 = traj[i+1]['observations']
     if (not np.array_equal(states_1[len(states_1)-1], states_2[0])):
         print(i)
         #break

    






"""

data = get_dataset('kitchen', 'partial')
traj = data.get_trajectories()
actions = traj[329]['actions']
#get environment
env, d_s, d_a= get_env('kitchen', 'partial')

rewards = []

#start animation
set_seed(0)
env.reset()
frames = []
for i in range(len(actions)):
    #action = np.random.uniform(-1.0, 1.0, d_a)
    #action = np.clip(actions[i], -1.0, 1.0)
    obs, rew, terminated, truncated, info = env.step(actions[i])
    rewards.append(rew)
    frames.append(env.render())
    if terminated or truncated:
        break
#print(rewards)
media.write_video("demo.mp4", frames, fps=30)


"""



















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

