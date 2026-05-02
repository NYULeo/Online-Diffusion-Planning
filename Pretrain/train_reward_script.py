import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Rewards.Reward_Backbone import train_reward, test_Model, train_reward_pos_weight
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



"""
if __name__ == '__main__':
    set_seed(1)
    train_reward(
    dataset_name = 'pointmaze',
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
    sigma = None,
    alpha = 0.999,
    #alpha = None,
    target_reward = 25.0,
    specific_dataset = 'large',
    goal = np.array([[4.0, -3.0]], dtype = np.float32))
"""



if __name__ == '__main__':
    set_seed(1)
    trajs = check_trajs_exit('cube', 'single-play', 1, 0)
    new_trajs = trajs.copy()
    train_reward(
        dataset_name = 'cube',
        hidden_layers = 2,
        hidden_dim = 128, 
        batch_size = 256,
        num_steps = 500,
        save_freq = 500,
        lr = 1e-04,
        sigma = 8.0,
        #alpha = 0.99,
        target_reward = 50.0,
        specific_dataset = 'single',
        task_id = 1,
        traj_length = None,
        trajs = new_trajs
    )

    test_Model(
        dataset_name = 'cube', 
        hidden_layers = 2, 
        hidden_dim = 128,
        specific_dataset = 'single', 
        trajs = new_trajs,
        sigma = 8.0,
        #alpha = 0.99, 
        target_reward = 50.0,
        task_id = 1,
        traj_length = None,
        save_freq = 500, 
        num_steps = 500)








