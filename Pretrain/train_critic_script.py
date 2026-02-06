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



def get_trajs(env_name: str, specific_env: str, step: int):
    path = (
        REPO_ROOT
        / "Finetuning"
        / "Rollouts"
        / env_name
        / specific_env
        / f"Generated_trajs_Info_{step}.pkl"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find rollout pickle:\n  {path}\n\n"
            f"Generate it first (Finetuning rollout) or check env/specific/step."
        )
    with open(path, "rb") as f:
        trajs = pickle.load(f)
    return trajs



if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    env_name = 'pointmaze'
    specific_env = 'medium'
    data = get_dataset(env_name, specific_env)
    trajs = data.get_trajectories()

    
    half_trajs_1 = trajs[:int(len(trajs)*0.2)]
    trajs = get_trajs(env_name, specific_env, 50)
    half_trajs_2 = trajs[int(len(trajs)*0.2):]
    trajs = half_trajs_1 + half_trajs_2
    
    #trajs = get_trajs(env_name, specific_env, 30)
    
    
    
    """
    #large
    train_critic(dataset_name = env_name, 
                 specific_dataset = specific_env, 
                 sigma = 15.0, 
                 batch_size = 128, 
                 num_steps = 5000, 
                 gamma = 1.0, 
                 horizon = 32, 
                 lr = 1e-05, 
                 tau = 0.005,
                 goal = np.array([[4.0, -3.0]], dtype = np.float32),
                 target_reward = 10.0,
                 trajs = trajs)
    

    train_critic(dataset_name = env_name, 
                 specific_dataset = specific_env, 
                 sigma = 7.0, 
                 batch_size = 128, 
                 num_steps = 55000, 
                 gamma = 1.0, 
                 horizon = 32, 
                 lr = 1e-05, 
                 tau = 0.005,
                 goal = np.array([[-4.0, -3.0]], dtype = np.float32),
                 target_reward = 1.0,
                 trajs = trajs)
    """

    """
    test_critic(dataset_name = env_name, 
                specific_dataset = specific_env, 
                checkpoint_step = 100000, 
                sigma = 7.0, 
                gamma = 0.99, 
                horizon = 32, 
                goal =  np.array([[-2.5, -2.5]], dtype = np.float32),
                target_reward = 1.0, 
                trajs = trajs)
     """
     #medium
    
    
    train_critic(dataset_name = env_name, 
                 specific_dataset = specific_env, 
                 sigma = 7.0, 
                 batch_size = 256, 
                 num_steps = 3000, 
                 gamma = 0.99, 
                 horizon = 32, 
                 lr = 1e-05, 
                 tau = 0.005,
                 goal = np.array([[-2.5, -2.5]], dtype = np.float32),
                 target_reward = 1.0,
                 trajs = trajs)
    
    print('training complete')
    
    




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
 

