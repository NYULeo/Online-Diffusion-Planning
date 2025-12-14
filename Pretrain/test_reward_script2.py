
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Pretrain.Rewards.Reward_Backbone import test_Model
from utils import set_seed
import pickle
import numpy as np
import copy
from Pretrain.Rewards.Reward_Backbone import Train_Dataset, test_dataset
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d, convolve
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == '__main__':
    with open('./Pretrain/Rollouts/pointmaze/medium/Generated_trajs_Info.pkl', 'rb') as f:
         trajs_info = pickle.load(f)
    trajs = trajs_info['trajs']
    set_seed(1)
    test_Model(
    dataset_name = 'pointmaze', 
    specific_dataset = 'medium', 
    trajs = trajs,
    sigma = 100.0, 
    target_reward = 50.0,
    save_freq = 500, 
    num_steps = 5000)



"""
def boost_signal(target_reward, rews):
        for t in range(len(rews)):
            if(rews[t] == 1):
                 rews[t] = target_reward
        return rews

with open('Rollouts/pointmaze/medium/Generated_trajs_Info.pkl', 'rb') as f:
         trajs_info = pickle.load(f)
trajs = trajs_info['trajs']

_, reward_name, obs_dim, act_dim = Train_Dataset('pointmaze', 'medium')
dataset = test_dataset(trajs, 10, reward_name, 1000.0)
dataloader = DataLoader(dataset, batch_size = 1)
total_reward = 0.0
for s, a, r in dataloader:
    total_reward += r.item()
avg_reward = total_reward / len(dataloader)
print(f"Average Reward: {avg_reward:.4f}")





L = []
for i in range(len(trajs)):
     reward = trajs[i]['rewards']
     new_reward = boost_signal(1.0, reward)
     new_reward = gaussian_filter1d(new_reward, 100)
     L.append(new_reward)
     #print(np.mean(new_reward))





x = np.arange(len(L[0]))
y = L[0]
plt.figure(figsize=(8,4))
plt.bar(x, y, color='skyblue', edgecolor='blue')
plt.xlabel("Index position")
plt.ylabel("Value at position")
plt.title("Value vs Index position")
plt.xticks(x)   # show all indices
plt.show()

y = L[0]
print(np.var(y))
x = np.arange(len(y))

plt.figure(figsize=(8,4))
plt.plot(x, y,  linestyle='-', color='blue', linewidth=2)  # line + markers
plt.xlabel("Index position")
plt.ylabel("Value at position")
plt.title("Value vs Index position (curve)")
plt.xticks(x)
plt.grid(True)
plt.show()

#marker='o',


my_list = L[0]# replace with your data
plt.figure(figsize=(8, 4))
plt.hist(my_list, bins=10, edgecolor='black')  # you can choose bins
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Values")
plt.show()
"""


