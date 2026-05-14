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

def check_cube_double_goal_reach(trajs, task_id):   
    goals = {   'task_1': [np.array([0.00000000e+00, 4.40762988e-19, 1.99598996e-01]),  np.array([0.0,   1.0,   0.199599])], 
                'task_2': [np.array([-0.75,      1.0,        0.199599]),  np.array([0.75,     1.0,       0.199599])],
                'task_3': [np.array([0.0,       -2.0,        0.199599]),  np.array([0.0,      2.0,       0.199599])],
                'task_4': [np.array([0.0,        1.0,        0.199599]),  np.array([0.0,       -1.0,        0.199599])],
                'task_5': [np.array([0.00000000e+00,  -3.99397428e-18,   1.99213779e-01]),  np.array([0.00000000e+00,   9.37726514e-18,   5.99039293e-01])]     }
    total_dist = 0.0
    for traj in trajs:
           position_1 = traj['observations'][-1][19:22]
           position_2 = traj['observations'][-1][28:31]
           dist_1 = np.linalg.norm(position_1 - goals[f"task_{task_id}"][0])
           dist_2 = np.linalg.norm(position_2 - goals[f"task_{task_id}"][1])
           total_dist += dist_1 + dist_2
    average_dist = total_dist/len(trajs)
    print(f"Task {task_id} average distance: {average_dist}")








if __name__ == '__main__':  # pragma: no cover
    
    set_seed(1)
    env_name = 'cube'
    specific_env = 'single-play'
    task_id = 4
    traj_length = 200



    data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
    trajs_1 = data.get_trajectories()
    path = PROJECT_ROOT / "Finetuning" / "Rollouts" / env_name / specific_env /f"task_{task_id}" / f"trajs_task{task_id}_success_0.pkl"
    with open(path, 'rb') as f:
          trajs_2 = pickle.load(f)
    
    
    trajs =  trajs_1 + trajs_2
    if(specific_env == 'double-play'):
        check_cube_double_goal_reach(trajs, task_id)
    else:
        check_cube_single_goal_reach(trajs, task_id)
   
    train_critic(dataset_name = env_name,
                 specific_dataset = specific_env, 
                 hidden_layers = 4,
                 hidden_dim = 512,
                 batch_size = 256, 
                 num_steps = 80000, 
                 gamma = 0.99, 
                 horizon = 32, 
                 lr = 1e-05,
                 min_lr = 5e-06,
                 tau = 0.005,
                 goal = None,
                 sigma = 3.0,
                 #alpha = None,
                 #alpha = None,
                 target_reward = 50.0,
                 trajs = trajs, 
                 task_id = task_id)
    
    data = get_dataset(env_name, specific_env, task_id = task_id, traj_length = traj_length)
    trajs_1 = data.get_trajectories()
    path = PROJECT_ROOT / "Finetuning" / "Rollouts" /  env_name / specific_env / f"task_{task_id}" / f"trajs_task{task_id}_success_0.pkl"
    with open(path, 'rb') as f:
          trajs_2 = pickle.load(f)
    trajs =  trajs_1 + trajs_2
    if(specific_env == 'double-play'):
        check_cube_double_goal_reach(trajs, task_id)
    else:
        check_cube_single_goal_reach(trajs, task_id)




    test_critic(dataset_name = env_name,
                specific_dataset = specific_env,
                hidden_layers = 4,
                hidden_dim = 512,
                checkpoint_step = 80000,
                gamma = 0.99,
                horizon = 32,
                goal = None,
                sigma = 3.0,
                #alpha = 0.99,
                target_reward = 50.0,
                trajs = trajs,
                task_id = task_id)
