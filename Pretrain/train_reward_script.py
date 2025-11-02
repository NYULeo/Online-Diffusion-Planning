from Rewards.Reward_Backbone import train_reward
from utils import set_seed



if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'pointmaze',
    batch_size=256, 
    num_steps=100000,   
    lr=1e-4,
    sigma=50,
    target_reward=1.0,
    specific_dataset='medium')




