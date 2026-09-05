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
    
    dataset_name = 'cube'
    specific_dataset = 'single'
    task_id = 4
    traj_length = None
    
   

    train_reward(dataset_name = dataset_name, 
                 hidden_layers = 4, 
                 hidden_dim = 512, 
                 batch_size = 256, 
                 num_steps = 30000, 
                 save_freq = 30000, 
                 lr = 5e-03, 
                 min_lr = 5e-04, 
                 sigma  = 4.0,
                 #sigma = None,
                 alpha = None, 
                 target_reward = 500.0,
                 specific_dataset = specific_dataset, 
                 task_id = task_id, 
                 traj_length = traj_length)
    

  
    test_Model(dataset_name, 
               hidden_layers = 4, 
               hidden_dim = 512, 
               specific_dataset = specific_dataset, 
               trajs = None, 
               sigma = 4.0, 
               #sigma = None,
               alpha = None, 
               target_reward = 500.0, 
               task_id = task_id,
               traj_length = traj_length, 
               save_freq = 30000, 
               num_steps = 30000)
"""


if __name__ == '__main__':
    import wandb

    set_seed(1)

    dataset_name = 'cube'
    specific_dataset = 'single'
    task_id = 4
    traj_length = None
    hp = {
        "dataset_name": dataset_name,
        "specific_dataset": specific_dataset,
        "task_id": task_id,
        "traj_length": traj_length,
        "hidden_layers": 4,
        "hidden_dim": 512,
        "batch_size": 256,
        "num_steps": 20000,
        "save_freq": 20000,
        "lr": 5e-03,
        "min_lr": 5e-05,
        "sigma": 3.0,
        "alpha": None,
        "target_reward": 50.0,
    }

    # Initialize wandb
    wandb.init(
        entity="kaiwen_hu-uc-berkeley",
        project="ODP",
        name=f"{dataset_name}-{specific_dataset}-task{task_id}-reward",
        config=hp,
    )

    train_keys = (
        "dataset_name", "hidden_layers", "hidden_dim", "batch_size",
        "num_steps", "save_freq", "lr", "min_lr", "sigma", "alpha",
        "target_reward", "specific_dataset", "task_id", "traj_length",
    )
    train_reward(**{k: hp[k] for k in train_keys})


    test_keys = (
        "dataset_name", "hidden_layers", "hidden_dim", "specific_dataset",
        "sigma", "alpha", "target_reward", "task_id", "traj_length",
        "save_freq", "num_steps",
    )
    test_Model(trajs=None, **{k: hp[k] for k in test_keys})
    wandb.finish()


"""
if __name__ == '__main__':
    set_seed(1)
    
    dataset_name = 'cube'
    specific_dataset = 'double'
    task_id = 4
    traj_length = None
    
   
    
    train_reward(dataset_name = dataset_name, 
                 hidden_layers = 4, 
                 hidden_dim = 512, 
                 batch_size = 256, 
                 num_steps = 30000, 
                 save_freq = 30000, 
                 lr = 5e-03, 
                 min_lr = 5e-04, 
                 #sigma  = 3.0,
                 sigma = None,
                 alpha = None, 
                 #target_reward = 10.0,
                 target_reward = None,
                 specific_dataset = specific_dataset, 
                 task_id = task_id, 
                 traj_length = traj_length)
    
       

  
    test_Model(dataset_name, 
               hidden_layers = 4, 
               hidden_dim = 512, 
               specific_dataset = specific_dataset, 
               trajs = None, 
               #sigma = 3.0, 
               sigma = None,
               alpha = None, 
               #target_reward = 10.0, 
               target_reward = None, 
               task_id = task_id,
               traj_length = traj_length, 
               save_freq = 30000, 
               num_steps = 30000)

"""




