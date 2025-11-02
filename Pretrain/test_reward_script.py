from Rewards.Reward_Backbone import test_Model, test_dataset, save_model
from Rewards.nets import LargeScalarReward
from Dataset import get_dataset
from utils import set_seed
import pickle
import numpy as np
import copy

"""
if __name__ == '__main__':
    with open('./Rollouts/kitchen/partial/Generated_trajs_Info.pkl', 'rb') as f:
         trajs_info = pickle.load(f)
    Gen_trajs = trajs_info['trajs']
    

"""







