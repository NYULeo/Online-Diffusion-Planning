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
    num_steps = 50000, 
    save_freq = 5000,  
    lr = 3e-4,
    sigma = 10.0,
    target_reward = 1.0,
    specific_dataset='medium')


