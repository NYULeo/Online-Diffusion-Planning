import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import argparse
import pickle
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Rewards.Reward_Backbone import train_reward, test_Model, train_reward_pos_weight
from Pretrain.utils import set_seed
from Pretrain.Critic.train_critic import train_critic
from Pretrain.utils import set_seed
from Pretrain.Dataset import get_dataset
from Pretrain.Critic.train_critic import test_critic


def check_cube_single_goal_reach(trajs, task_id):   
    goals = {'task_1': np.array( [ 0.0,       -1.0,        0.199599]), 
         'task_2': np.array([7.50000000e-01, 8.02418254e-18, 1.99598996e-01]),
         'task_3': np.array([-7.50000000e-01,  1.21832368e-19,  1.99598996e-01]),
         'task_4': np.array([0.75,     2.0,       0.199599]),
         'task_5': np.array([ 0.75,     -2.0,        0.199599])}
    
    total_dist = 0.0
    for traj in trajs:
           position = traj['observations'][-1][19:22]
           total_dist += np.linalg.norm(position - goals[f"task_{task_id}"])
    average_dist = total_dist/len(trajs)
    print(f"Task {task_id} average distance: {average_dist}")








if __name__ == '__main__':  # pragma: no cover
    
    set_seed(1)
    env_name = 'cube'
    specific_env = 'single-play'
    data = get_dataset(env_name, specific_env, task_id = 4, traj_length = None)
    trajs_1 = data.get_trajectories()
    path = PROJECT_ROOT / "Finetuning" / "Rollouts" / "cube" / "single-play" / "task_4" / "trajs_task4_success_0.pkl"
    with open(path, 'rb') as f:
          trajs_2 = pickle.load(f)
    trajs = trajs_2
    check_cube_single_goal_reach(trajs, 4)
    train_critic(dataset_name = env_name,
                 specific_dataset = specific_env, 
                 hidden_layers = 4,
                 hidden_dim = 512,
                 batch_size = 256, 
                 num_steps = 30000, 
                 gamma = 0.99, 
                 horizon = 32, 
                 lr = 3e-04, 
                 tau = 0.005,
                 goal = None,
                 sigma = 8.0,
                 #alpha = 0.99,
                 #alpha = None,
                 target_reward = 50.0,
                 trajs = trajs, 
                 task_id = 4)
    
    data = get_dataset(env_name, specific_env, task_id = 4, traj_length = None)
    trajs_1 = data.get_trajectories()
    
  
    path = PROJECT_ROOT / "Finetuning" / "Rollouts" / "cube" / "single-play" / "task_4" / "trajs_task4_success_0.pkl"

    with open(path, 'rb') as f:
          trajs_2 = pickle.load(f)
    trajs =  trajs_2
    check_cube_single_goal_reach(trajs, 4)
    test_critic(dataset_name = env_name,
                specific_dataset = specific_env,
                hidden_layers = 4,
                hidden_dim = 512,
                checkpoint_step = 30000,
                gamma = 0.99,
                horizon = 32,
                goal = None,
                sigma = 8.0,
                #alpha = 0.99,
                target_reward = 50.0,
                trajs = trajs,
                task_id = 4)

   

    
    
    
   
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    #large
    set_seed(1)
    env_name = 'pointmaze'
    specific_env = 'large'
    data = get_dataset(env_name, specific_env)
    trajs = data.get_trajectories()
    train_critic(dataset_name = env_name,
                 specific_dataset = specific_env, 
                 hidden_layers = 3,
                 hidden_dim = 128,
                 batch_size = 512, 
                 num_steps = 10000, 
                 gamma = 0.99, 
                 horizon = 20, 
                 lr = 3e-05, 
                 tau = 0.005,
                 goal = np.array([[4.0, -3.0]], dtype = np.float32),
                 #sigma = 7.0,
                 alpha = 0.999,
                 #alpha = None,
                 target_reward = 25.0,
                 trajs = trajs)
    

    
    
    test_critic(dataset_name = env_name, 
                specific_dataset = specific_env, 
                hidden_layers = 3, 
                hidden_dim = 128,
                checkpoint_step = 10000, 
                #sigma = 300.0, 
                gamma = 0.99, 
                horizon = 20, 
                goal =  np.array([[4.0, -3.0]], dtype = np.float32),
                alpha = 0.999,
                target_reward = 25.0, 
                trajs = trajs)
    """
    
    
    #medium
    """
    train_critic(dataset_name = env_name, 
                 specific_dataset = specific_env, 
                 sigma = 7.0, 
                 batch_size = 256, 
                 num_steps = 5000, 
                 gamma = 0.95, 
                 horizon = 32, 
                 lr = 1e-05, 
                 tau = 0.005,
                 goal = np.array([[-2.5, -2.5]], dtype = np.float32),
                 target_reward = 20.0,
                 trajs = trajs)
    print('training complete')
   
    
    test_critic(dataset_name = env_name, 
                specific_dataset = specific_env, 
                checkpoint_step = 5000, 
                sigma = 7.0, 
                gamma = 0.95, 
                horizon = 32, 
                goal =  np.array([[-2.5, -2.5]], dtype = np.float32),
                target_reward = 20.0, 
                trajs = trajs)
    """
    








    """
    train_critic(dataset_name = env_name, 
                 specific_dataset = specific_env, 
                 sigma = 7.0, 
                 batch_size = 256, 
                 num_steps = 5000, 
                 gamma = 1.0, 
                 horizon = 32, 
                 lr = 1e-05, 
                 tau = 0.005,
                 goal = np.array([[-2.5, -2.5]], dtype = np.float32),
                 target_reward = 1.0,
                 trajs = trajs)
    
    print('training complete')
    """
 

