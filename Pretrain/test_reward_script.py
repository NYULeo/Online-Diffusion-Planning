from Rewards.Reward_Backbone import test_Model, test_dataset, save_model
from Dataset import get_dataset
from utils import set_seed
import pickle
import numpy as np
import copy


if __name__ == '__main__':
    set_seed(1)
    test_Model(
    dataset_name = 'cube', 
    hidden_layers = 4, 
    hidden_dim = 256,
    specific_dataset = 'single', 
    trajs = None,
    sigma = None,
    alpha = 0.999, 
    target_reward = 25.0,
    task_id = 1,
    save_freq = 1000, 
    num_steps = 1000)
    






