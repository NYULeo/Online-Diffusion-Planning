from Reward_Backbone import train_reward
from utils import set_seed



if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'kitchen',  
    batch_size=1024, 
    num_steps=500,   
    lr=1e-4,
    sigma=3)