import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
import torch
import numpy as np
import mediapy as media
from Pretrain.Dataset import get_env
from Pretrain.Planners.Backbone.Dit import DiT1d
#from Pretrain.Planners.Backbone.utils import get_pretrained_planner
from utils import get_planner, get_normalized_score
from Pretrain.Dataset import Planner_Processor
from Pretrain.Planners.Backbone.Sampler import sample_reverse_sde, sample_euler_karras, sample_euler_karras2
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv 
import pickle
import random
import gymnasium as gym
import gymnasium_robotics
from Pretrain.Dataset import get_dataset
from typing import Optional
from utils import get_normalized_score, rollout_parallel


def set_seed(seed=0):
    # Python random
    random.seed(seed)
    # NumPy random
    np.random.seed(seed)
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    # PyTorch deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_trajs(trajs, env_name, specific_env):
    os.makedirs(f'./Finetuning/Rollouts/{env_name}/{specific_env}/', exist_ok=True)
    save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info.pkl'
    with open(save_path, 'wb') as f:
         pickle.dump(trajs, f)
    print(f"trajectories saved")

def rollout(env_name, specific_env, horizon, steps_T, num_karras, eta, episode_length, checkpoint_steps, render = False, goal_cell: Optional[np.ndarray] = None, start_cell: Optional[np.ndarray] = None, base_seed: int = 0):
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, Checkpoint_steps; {checkpoint_steps}")
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     device = "cuda" if torch.cuda.is_available() else "cpu"
     print(f"Using device {device}")
     
     env, d_s, d_a = get_env(env_name, specific_env, render_mode = 'rgb_array')
    
    # Create environment factory function
     state_dict = get_planner(env_name, specific_env, checkpoint_steps)
     if( env_name == 'kitchen'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif (env_name == 'pointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     else:
          raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()

     #get Processor
     planner_processor = Planner_Processor(env_name, specific_env)

     #reset
     #s0 = env.reset(seed = 0, options={"goal_cell": goal_cell, "reset_cell": reset_cell})
     if(goal_cell is not None):
        s0 = env.reset(seed = base_seed, options={"goal_cell": goal_cell, "reset_cell": start_cell})
     else:
        s0 = env.reset(seed = base_seed)
     s0 = s0[0]['observation']
     current_state = s0
     frames = []
     observations = []
     actions = []
     rewards = []
     for i in range(episode_length):
           current_state_norm = planner_processor.preprocess(current_state)
           
           #x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta,  device = device)
           x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
           action = x[0, d_s:(d_s+d_a)].copy()
           obs, reward, terminated, truncated, info = env.step(action)
           if(render):
                frames.append(env.render())
                
           
           observations.append(obs['observation'].copy())
           actions.append(action.copy())
           rewards.append(reward)
           current_state = obs['observation'].copy()
           #print(f"Episode {i} reward: {reward}")
           if(terminated or truncated):
                #print(f"Episode {i} terminated or truncated")
                break
     
     env.close()
     traj = {'observations': np.asarray(observations), 'actions': np.asarray(actions), 'rewards': np.asarray(rewards)}
     traj_info = {'sequence': traj, 'env_name': env_name, 'specific_env': specific_env }
     #print(rewards)
     if(render):
          media.write_video("demo.mp4", frames, fps=50) #save the video
     """
     with open('Generated_trajectory.pkl', 'wb') as f:
                pickle.dump(traj_info, f)
     """
     
     #print(get_normalized_score([traj]))


    



# ---- 4) Example usage (fill ScoreWrapper first) ----
if __name__ == "__main__":
    set_seed(0)
    horizon = 32
    env_name = 'kitchen'
    specific_train_dataset = 'partial'
    #rollout(env_name, specific_train_dataset, horizon, steps_T = 50, num_karras = 3, eta = 0.8, episode_length = 4000, checkpoint_steps = 70, render = True,  goal_cell = np.array([6, 1], dtype = int), start_cell = np.array([5, 4], dtype = int))
    #rollout(env_name, specific_train_dataset, horizon, steps_T = 150, num_karras = 8, eta = 0.8, episode_length = 4000, checkpoint_steps = 0, render = True, base_seed = 10)
    #150, 8
    #50, 3
    rollout_parallel(env_name, specific_train_dataset, horizon = 32, steps_T = 150, num_karras = 8, eta = 0.8, episode_length = 4000, checkpoint_step = 0, num_envs = 4, seed_base = 0)
    """
    checkpoint = 0
    while(checkpoint < 450):
       print(f"Rollout for checkpoint: {checkpoint}")
       rollout_parallel(env_name,  specific_train_dataset, horizon = 32, steps_T = 50, num_karras = 3, eta = 0.8, episode_length = 1000, critic = False, checkpoint_steps = checkpoint, num_envs=8)
       checkpoint += 50
    """
    



"""
env, d_s, d_a = get_env('pointmaze', 'medium')

maze = env.unwrapped.maze  # Access the internal Maze object
maze_map = maze.maze_map
rows, cols = len(maze_map), len(maze_map[0])
    
# Find all free cells (not walls)
free_cells = []
for row in range(rows):
    for col in range(cols):
        if maze_map[row][col] != 1:  # 1 = wall; others are free/open
               free_cells.append(np.array([row, col]))
free_cells = np.array(free_cells)
print(free_cells)
"""

