from Reward_Backbone import test_Model
from utils import set_seed
import pickle
import numpy as np
import copy


if __name__ == '__main__':
    with open('Generated_trajectories.pkl', 'rb') as f:
        info = pickle.load(f)
    trajs = info['trajs']
    data = []
    for traj in trajs:
       obs = []
       acts = []
       rewards = []
       for step in traj:
            obs.append(step['observation'].copy())
            acts.append(step['action'].copy())
            rewards.append(step['reward'].copy())
       obs = np.array(obs)
       acts = np.array(acts)
       rewards = np.array(rewards)
       traj = {'observations': obs, 'actions': acts, 'rewards': rewards}
       data.append(traj)
    set_seed(1)
    test_Model(
    dataset_name = 'pointmaze', 
    specific_dataset = 'medium', 
    trajs = data,
    sigma = 3, 
    save_freq = 100, 
    num_steps = 10000)