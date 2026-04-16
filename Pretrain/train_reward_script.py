import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Rewards.Reward_Backbone import train_reward
from Pretrain.utils import set_seed
import numpy as np

"""
if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'pointmaze',
    batch_size = 256, 
    num_steps = 400, 
    save_freq = 200,  
    lr = 1e-4,
    sigma = 7.0,
    target_reward = 20.0,
    specific_dataset = 'medium',
    goal = np.array([[-2.5, -2.5]], dtype = np.float32))
"""

"""
if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'pointmaze',
    hidden_layers = 2,
    hidden_dim = 128,
    batch_size = 512, 
    num_steps = 1000, 
    save_freq = 500,  
    lr = 1e-04,
    sigma = None,
    alpha = 0.999,
    #alpha = None,
    target_reward = 25.0,
    specific_dataset = 'large',
    goal = np.array([[4.0, -3.0]], dtype = np.float32))
"""

if __name__ == '__main__':
    set_seed(1)
    train_reward(
        dataset_name = 'cube',
        hidden_layers = 4,
        hidden_dim = 256, 
        batch_size = 512,
        num_steps = 1000,
        save_freq = 1000,
        lr = 1e-04,
        sigma = None,
        alpha = 0.999,
        target_reward = 25.0,
        specific_dataset = 'single',
        task_id = 1
    )





