import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Rewards.Reward_Backbone import train_reward
from Pretrain.utils import set_seed
import numpy as np


if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'kitchen',
    batch_size = 128, 
    num_steps = 5000, 
    save_freq = 1000,  
    lr = 3e-4,
    sigma = 5.0,
    target_reward = 1.0,
    specific_dataset='partial',
    goal = None)


