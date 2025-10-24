from Rewards.Reward_Backbone import test_Model, test_dataset
from Dataset import get_dataset
from utils import set_seed
import pickle
import numpy as np
import copy


if __name__ == '__main__':
    with open('./Rollouts/pointmaze/medium/Generated_trajs_Info.pkl', 'rb') as f:
         trajs_info = pickle.load(f)
    Gen_trajs = trajs_info['trajs']
    
    """
    data1 = get_dataset('kitchen', 'complete')
    traj1 = data1.get_trajectories()
    data2 = get_dataset('kitchen', 'partial')
    traj2 = data2.get_trajectories()
    data3 = get_dataset('kitchen', 'mixed')
    traj3 = data3.get_trajectories()
    """

    
    print(f"Testing on the Generated Trajectories")
    set_seed(1)
    test_Model(
    dataset_name = 'pointmaze',
    specific_dataset = 'medium',
    trajs = Gen_trajs,
    sigma = 3, 
    save_freq = 100, 
    num_steps = 10000)
"""
    print(f"Testing on the partial Trajectories")
    set_seed(1)
    test_Model(
    dataset_name = 'kitchen', 
    trajs = traj2,
    sigma = 3, 
    save_freq = 100, 
    num_steps = 10000)

    print(f"Testing on the mixed Trajectories")
    set_seed(1)
    test_Model(
    dataset_name = 'kitchen', 
    trajs = traj3,
    sigma = 3, 
    save_freq = 100, 
    num_steps = 10000)

    print(f"Testing on the complete Trajectories")
    set_seed(1)
    test_Model(
    dataset_name = 'kitchen', 
    trajs = traj1,
    sigma = 3, 
    save_freq = 100, 
    num_steps = 10000)

"""
