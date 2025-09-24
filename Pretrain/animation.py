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

def check_rewards(trajs):
    print('Checking Rewards')
    for i in range(len(trajs)):
       D = {}
       rewards = trajs[i]['rewards']
       #print(len(rewards))
       for j in range(len(rewards)):
          if(rewards[j] in D.keys()):
               D[rewards[j]] = D[rewards[j]]+1
          else:
               D[rewards[j]] = 1
       count = True
       for key in D.keys():
           if(key > 0):
               count = False
               break
       print(D)
       if(count):
            print(i)

def check_speration(trajs):
    print('Checking Separation')
    for i in range(len(trajs)-1):
     states_1 = trajs[i]['observations']
     states_2 = trajs[i+1]['observations']
     if (np.array_equal(states_1[len(states_1)-1], states_2[0])):
         print(i)

def Rollout(env, actions):
     set_seed(0)
     env.reset()
     rewards = []
     frames = []
     for i in range(len(actions)):
         obs, rew, terminated, truncated, info = env.step(actions[i])
         rewards.append(rew)
         frames.append(env.render())
         if terminated or truncated:
             break
     media.write_video("demo.mp4", frames, fps=30)






data = get_dataset('AntMaze', 'umaze')
traj = data.get_trajectories()

check_rewards(traj)
check_speration(traj)






"""
data = get_dataset('kitchen', 'partial')
traj = data.get_trajectories()
actions = traj[0]['actions']
env, _, _ = get_env('kitchen', 'complete')
Rollout(env, actions)

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

