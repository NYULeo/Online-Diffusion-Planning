from Rewards.Reward_Backbone import train_reward
from utils import set_seed



if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'kitchen',  
    batch_size=1024, 
    num_steps=10000,   
    lr=1e-1,
    sigma=3)



