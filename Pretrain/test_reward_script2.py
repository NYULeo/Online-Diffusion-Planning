
from Rewards.Reward_Backbone import test_Model
from utils import set_seed
import pickle
import numpy as np
import copy


if __name__ == '__main__':
    with open('Rollouts/pointmaze/medium/Generated_trajs_Info.pkl', 'rb') as f:
         trajs_info = pickle.load(f)
    trajs = trajs_info['trajs']
    set_seed(1)
    test_Model(
    dataset_name = 'kitchen', 
    specific_dataset = 'partial', 
    trajs = trajs,
    sigma = 3, 
    target_reward = 5.0,
    save_freq = 50, 
    num_steps = 1000)
