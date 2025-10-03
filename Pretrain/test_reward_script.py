from Reward_Backbone import test_Model
from utils import set_seed


if __name__ == '__main__':
    set_seed(1)
    test_Model(
    dataset_name = 'pointmaze', 
    specific_dataset = 'medium', 
    sigma = 3, 
    save_freq = 100, 
    num_steps = 10000)