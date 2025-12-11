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
    num_steps = 500000, 
    save_freq = 50000,  
    lr = 1e-4,
    sigma = 100,
    target_reward = 50.0,
    specific_dataset='medium')


