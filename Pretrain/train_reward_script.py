import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Rewards.Reward_Backbone import train_reward
from Pretrain.utils import set_seed



if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'pointmaze',
    batch_size = 256, 
    num_steps = 100000, 
    save_freq = 10000,  
    lr = 1e-3,
    sigma = 3,
    target_reward = 1.0,
    specific_dataset='medium')


