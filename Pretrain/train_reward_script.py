import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Rewards.Reward_Backbone import train_reward, test_Model, train_reward_pos_weight, train_reward_ensemble, test_Model_ensemble
from Pretrain.utils import set_seed
import numpy as np
import pickle



def check_trajs_exit(env_name, specific_env, task_id, step):
    from pathlib import Path
    if(task_id is not None):
         path = Path(f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{task_id}/Generated_trajs_Info_{step}.pkl')
    else:
         path = Path(f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_{step}.pkl')
    if not path.exists():
        print(f"trajs not found")
        return None
    else:
        with path.open('rb') as f:
             trajs = pickle.load(f)
        return trajs

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



"""

if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'pointmaze',
    hidden_layers = 1,
    hidden_dim = 32,
    batch_size = 256, 
    num_steps = 400, 
    save_freq = 200,  
    lr = 1e-4,
    sigma = 7.0,
    target_reward = 20.0,
    specific_dataset = 'medium',
    goal = np.array([[-2.5, -2.5]], dtype = np.float32))

"""
"""
if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'pointmaze',
    hidden_layers = 2,
    hidden_dim = 128,
    batch_size = 512, 
    num_steps = 1000, 
    save_freq = 500,  
    lr = 1e-04,
    sigma = 2.0,
    target_reward = 20.0,
    specific_dataset = 'large',
    goal = np.array([[4.0, -3.0]], dtype = np.float32))

"""


if __name__ == '__main__':
    set_seed(1)
    
    dataset_name = 'cube'
    specific_dataset = 'single'
    task_id = 4
    traj_length = 200
    
    """
    path = f'./Finetuning/Rollouts/{dataset_name}/{specific_dataset}-play/task_{task_id}/trajs_task{task_id}_success_0.pkl'
    with open(path, 'rb') as f:
          trajs = pickle.load(f)
    """
    
    """
    train_reward_ensemble(
        dataset_name = dataset_name,
        hidden_layers = 3,
        hidden_dim = 256, 
        batch_size = 256,
        num_steps = 30000,
        save_freq = 5000,
        lr = 3e-04,
        min_lr = 3e-05,
        ensemble_size = 5,
        bootstrap = True,
        save_percentage = 0.02,
        #lr = 1e-04,
        #min_lr = 5e-06,
        sigma = 4.0,
        #sigma = None,
        #alpha = None,
        #alpha = 0.99,
        target_reward = 500.0,
        specific_dataset = specific_dataset,
        task_id = task_id,
        traj_length = traj_length,
        trajs = None,
        log_every = 2000
    )
    """
    
    train_reward(dataset_name = dataset_name, 
                 hidden_layers = 4, 
                 hidden_dim = 512, 
                 batch_size = 256, 
                 num_steps = 30000, 
                 save_freq = 10000, 
                 lr = 5e-03, 
                 min_lr = 5e-04, 
                 sigma = 4.0,
                 alpha = None, 
                 target_reward = 500,
                 specific_dataset = specific_dataset, 
                 task_id = task_id, 
                 traj_length = traj_length)
    
    """
    path = f'./Finetuning/Rollouts/{dataset_name}/{specific_dataset}-play/task_{task_id}/trajs_task{task_id}_success_0.pkl'
    with open(path, 'rb') as f:
          trajs = pickle.load(f)
    """
    """
    test_Model_ensemble(
        dataset_name = dataset_name, 
        hidden_layers = 3, 
        hidden_dim = 256,
        ensemble_size = 5,
        specific_dataset = specific_dataset, 
        trajs = None,
        sigma = 4.0,
        #sigma = None,
        #alpha = None,
        #alpha = 0.99, 
        target_reward = 500.0,
        task_id = task_id,
        traj_length = traj_length,
        save_freq = 30000, 
        num_steps = 30000)
    """
    test_Model(dataset_name, 
               hidden_layers = 4, 
               hidden_dim = 512, 
               specific_dataset = specific_dataset, 
               trajs = None, 
               sigma = 4.0, 
               alpha = None, 
               target_reward = 500, 
               task_id = 4,
               traj_length = traj_length, 
               save_freq = 10000, 
               num_steps = 30000)
    




"""

if __name__ == '__main__':
    set_seed(1)
    
    dataset_name = 'ogpointmaze'
    specific_dataset = 'medium'
    task_id = 1
    
    
    
    
    train_reward(
        dataset_name = dataset_name,
        hidden_layers = 1,
        hidden_dim = 32, 
        batch_size = 256,
        num_steps = 12000,
        save_freq = 2000,
        lr = 1e-04,
        min_lr = 1e-04,
        sigma = 7.0,
        #alpha = 0.99,
        target_reward = 50.0,
        specific_dataset = specific_dataset,
        task_id = task_id,
        traj_length = None,
        trajs = None
    )
    
    test_Model(dataset_name = dataset_name, 
               hidden_layers = 1, 
               hidden_dim = 32, 
               specific_dataset  = specific_dataset, 
               trajs = None, 
               sigma = 7.0, 
               alpha = None, 
               target_reward = 50.0,
               goal= None, 
               task_id = task_id, 
               traj_length = None, 
               save_freq = 2000, 
               num_steps = 12000)

"""
