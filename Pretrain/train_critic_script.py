import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Pretrain.Critic.train_critic import train_critic
from Pretrain.utils import set_seed
from Pretrain.Dataset import get_dataset
import numpy as np

if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    env_name = 'pointmaze'
    specific_env = 'large'
    data = get_dataset(env_name, specific_env)
    trajs = data.get_trajectories()
    #trajs = get_trajs(env_name, specific_env, 0)
    train_critic(dataset_name = env_name, 
                 specific_dataset = specific_env, 
                 sigma = 10.0, 
                 batch_size = 512, 
                 num_steps = 2000, 
                 gamma = 1.0, 
                 horizon = 32, 
                 lr = 5e-5, 
                 tau = 0.01,
                 goal = np.array([[4.5, 3.0]], dtype = np.float32),
                 target_reward = 1.0,
                 trajs = trajs)
    print('training complete')
    
    """
    trajs = data.get_trajectories()
    step = 2000
    while(step <= 10000):
        test_critic(dataset_name = env_name, 
                    specific_dataset = specific_env, 
                    checkpoint_step = step, 
                    sigma = 10.0, 
                    gamma = 1.0, 
                    horizon = 32, 
                    goal = None,
                    target_reward = 50.0, 
                    trajs = trajs)
        step += 2000
    print('testing complete')
    """
