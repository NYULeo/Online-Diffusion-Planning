from Reward_Backbone import test_reward
from utils import set_seed


if __name__ == '__main__':
    set_seed(1)
    test_reward(
    dataset_name = 'kitchen', 
    specific_dataset = None, 
    sigma = 3, 
    save_freq = 50, 
    num_steps = 1000)