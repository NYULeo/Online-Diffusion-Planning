import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Rewards.Reward_Backbone import train_reward
from Pretrain.utils import set_seed
import numpy as np

if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'pointmaze',
    batch_size = 256, 
    num_steps = 1000, 
    save_freq = 200,  
    lr = 1e-4,
    sigma = 7.0,
    target_reward = 1.0,
    specific_dataset='medium',
    goal = np.array([[2.5, 2.5]], dtype = np.float32))
