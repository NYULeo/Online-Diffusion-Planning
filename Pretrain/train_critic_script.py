import sys
import os
from pathlib import Path
import argparse
import pickle
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]  # Online-Diffusion-Planning/
sys.path.insert(0, str(REPO_ROOT))
from Pretrain.Critic.train_critic import train_critic
from Pretrain.utils import set_seed
from Pretrain.Dataset import get_dataset
from Pretrain.Critic.train_critic import test_critic








if __name__ == '__main__':  # pragma: no cover
    
    set_seed(1)
    env_name = 'cube'
    specific_env = 'single-play'
    data = get_dataset(env_name, specific_env, task_id = 4, traj_length = None)
    trajs_1 = data.get_trajectories()
    path = f'./Finetuning/Rollouts/cube/single-play/task_4/trajs_task4_success_0.pkl'
    with open(path, 'rb') as f:
          trajs_2 = pickle.load(f)
    trajs = trajs_1 + trajs_2
    
    train_critic(dataset_name = env_name,
                 specific_dataset = specific_env, 
                 hidden_layers = 4,
                 hidden_dim = 512,
                 batch_size = 256, 
                 num_steps = 20000, 
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
    path = f'./Finetuning/Rollouts/cube/single-play/task_4/trajs_task4_success_0.pkl'
    with open(path, 'rb') as f:
          trajs_2 = pickle.load(f)
    trajs = trajs_1 + trajs_2

    test_critic(dataset_name = env_name,
                specific_dataset = specific_env,
                hidden_layers = 4,
                hidden_dim = 512,
                checkpoint_step = 20000,
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
 

