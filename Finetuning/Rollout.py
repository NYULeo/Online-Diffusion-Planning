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
from Finetuning.utils import get_planner, get_normalized_score, get_expert_score, spare_reward_prcocessor
from Pretrain.Dataset import Planner_Processor
from Pretrain.Planners.Backbone.Sampler import sample_reverse_sde, sample_euler_karras, sample_euler_karras2
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv 
import pickle
import random
import gymnasium as gym
import gymnasium_robotics
from Pretrain.Dataset import get_dataset
from typing import Optional
from utils import get_normalized_score, rollout_parallel, get_current_state



def test_rollout_fit_for_model(traj, dataset_name=None, specific_dataset=None, 
                                reward_checkpoint=0, kernel_checkpoint=0, 
                                critic_checkpoint=0, device=None):
    """
    Calculate average log probability, average reward, and average critic value 
    for a trajectory using the reward, kernel, and critic models.
    
    Args:
        traj: Trajectory dictionary with 'observations', 'actions', and 'rewards'
        dataset_name: Name of the dataset (e.g., 'kitchen', 'pointmaze')
        specific_dataset: Specific dataset variant (e.g., 'partial', 'medium')
        reward_checkpoint: Checkpoint step for reward model
        kernel_checkpoint: Checkpoint step for kernel model
        critic_checkpoint: Checkpoint step for critic model
        device: torch device (defaults to cuda if available, else cpu)
    
    Returns:
        dict: {
            'avg_log_prob': float,    # Average log probability from kernel model
            'avg_reward': float,      # Average reward from reward model
            'avg_critic': float       # Average critic value from critic model
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_name is None or specific_dataset is None:
        raise ValueError("dataset_name and specific_dataset must be provided")
    
    # Load reward model and stats
    from Finetuning.utils import (get_reward_model, get_reward_stats, get_kernel, 
                                   get_kernel_stats, get_critic_model, get_critic_stats)
    from Pretrain.Rewards.nets import SimpleReward
    from Pretrain.Transition_Kernel.Kernel_Net import RobustTransitionKernel
    from Pretrain.Critic.nets import Critic
    from Pretrain.Dataset import get_env
    
    reward_state_dict, obs_dim, act_dim = get_reward_model(dataset_name, specific_dataset, reward_checkpoint)
    reward_net = SimpleReward(obs_dim, act_dim).to(device)
    reward_net.load_state_dict(reward_state_dict)
    reward_net.eval()
    reward_stats = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)
    
    # Load kernel models and stats
    kernel_state_dicts, _, _ = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
    kernels = []
    for kernel_state_dict in kernel_state_dicts:
        kernel_net = RobustTransitionKernel(obs_dim, act_dim).to(device)
        kernel_net.load_state_dict(kernel_state_dict)
        kernel_net.eval()
        kernels.append(kernel_net)
    kernel_stats = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)
    
    # Load critic model and stats
    critic_state_dict, critic_obs_dim = get_critic_model(dataset_name, specific_dataset, critic_checkpoint)
    critic_net = Critic(critic_obs_dim).to(device)
    critic_net.load_state_dict(critic_state_dict)
    critic_net.eval()
    critic_stats = get_critic_stats(dataset_name, specific_dataset, critic_checkpoint)
    
    observations = traj['observations']
    actions = traj['actions']
    
    # Calculate average log probability, average reward, and average critic value
    total_log_prob = 0.0
    total_reward = 0.0
    total_critic = 0.0
    num_transitions = len(actions)
    num_states = len(observations)
    
    with torch.no_grad():
        for t in range(num_transitions):
            # Get state, action, and next state
            s = observations[t]
            a = actions[t]
            
            # Compute reward
            s_norm_reward = reward_stats.norm_obs(s)
            s_tensor = torch.tensor(s_norm_reward, dtype=torch.float32, device=device).unsqueeze(0)
            a_tensor = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
            r = reward_net(s_tensor, a_tensor)
            total_reward += r.item()
            
            # Compute critic value for current state
            # For pointmaze, critic uses only first 2 dimensions
            if dataset_name == 'pointmaze':
                s_critic = s[:2]
            else:
                s_critic = s
            s_norm_critic = critic_stats.norm_obs(s_critic)
            s_critic_tensor = torch.tensor(s_norm_critic, dtype=torch.float32, device=device).unsqueeze(0)
            v = critic_net(s_critic_tensor)
            total_critic += v.item()
            
            # Skip if we don't have next state for log prob calculation
            if t >= len(observations) - 1:
                continue
            
            s_next = observations[t + 1]
            
            # Compute log probability using kernel ensemble
            s_norm_kernel = kernel_stats.norm_obs(s)
            s_next_norm_kernel = kernel_stats.norm_obs(s_next)
            
            s_tensor = torch.tensor(s_norm_kernel, dtype=torch.float32, device=device).unsqueeze(0)
            a_tensor = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
            s_next_tensor = torch.tensor(s_next_norm_kernel, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Average log prob across ensemble
            ensemble_log_probs = []
            for kernel in kernels:
                mu, log_std = kernel(s_tensor, a_tensor)
                lp = kernel.log_prob(s_next_tensor, mu, log_std)
                ensemble_log_probs.append(lp.item())
            
            avg_log_prob_transition = np.mean(ensemble_log_probs)
            total_log_prob += avg_log_prob_transition
        
        # Compute critic value for the last state (if not already computed)
        if num_states > num_transitions:
            s_final = observations[num_states - 1]
            if dataset_name == 'pointmaze':
                s_final_critic = s_final[:2]
            else:
                s_final_critic = s_final
            s_final_norm_critic = critic_stats.norm_obs(s_final_critic)
            s_final_critic_tensor = torch.tensor(s_final_norm_critic, dtype=torch.float32, device=device).unsqueeze(0)
            v_final = critic_net(s_final_critic_tensor)
            total_critic += v_final.item()
    
    # Calculate averages
    # For log prob, we have num_transitions-1 transitions (last step has no next state)
    num_transitions_for_log_prob = num_transitions - 1 if num_transitions > 0 else 0
    avg_log_prob = total_log_prob / num_transitions_for_log_prob if num_transitions_for_log_prob > 0 else 0.0
    avg_reward = total_reward / num_transitions if num_transitions > 0 else 0.0
    avg_critic = total_critic / num_states if num_states > 0 else 0.0
    
    return {
        'avg_log_prob': avg_log_prob,
        'avg_reward': avg_reward,
        'avg_critic': avg_critic
    }

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
     print(f"Horizon: {horizon}, step_T: {steps_T}, num_karras: {num_karras}, eta: {eta}, Checkpoint_steps; {checkpoint_steps}, episode_length: {episode_length}")
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
     elif(env_name == 'antmaze'):
           model = DiT1d(in_dim = (d_s), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
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
     """
     if(env_name == 'antmaze'):
          current_state = np.concatenate([
               s0[0]['observation'],
               s0[0]['achieved_goal']
           ])
     else:
         current_state = s0[0]['observation']
     """
     current_state = get_current_state(s0[0], env_name)
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
                
           current_state = get_current_state(obs.copy(), env_name)
           observations.append(current_state.copy())
           actions.append(action.copy())
           rewards.append(reward)
           #current_state = obs['observation'].copy()
           #print(f"Episode {i} reward: {reward}")
           if(terminated or truncated):
                #print(f"Episode {i} terminated or truncated")
                break
     print(len(rewards))
     env.close()
     traj = {'observations': np.asarray(observations), 'actions': np.asarray(actions), 'rewards': np.asarray(spare_reward_prcocessor(rewards))}
     traj_info = {'sequence': traj, 'env_name': env_name, 'specific_env': specific_env }
     #print(test_rollout_fit_for_model(traj, env_name, specific_env, checkpoint_steps, checkpoint_steps, checkpoint_steps, device=None))
     
     #expert_score = get_expert_score(env_name)
     #print(get_normalized_score([traj], expert_score))
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
    env_name = 'pointmaze'
    specific_train_dataset = 'large'
    rollout(env_name, specific_train_dataset, horizon, steps_T = 50, num_karras = 3, eta = 0.8, episode_length = 10000, checkpoint_steps = 0, render = True,  goal_cell = np.array([1, 10], dtype = int), start_cell = np.array([7, 1], dtype = int))
    #rollout(env_name, specific_train_dataset, horizon, steps_T = 150, num_karras = 8, eta = 0.8, episode_length = 1000, checkpoint_steps = 0, render = True, base_seed = 0)
    #150, 8
    #50, 3
    #rollout_parallel(env_name, specific_train_dataset, horizon = 32, steps_T = 50, num_karras = 3, eta = 0.8, episode_length = 4000, checkpoint_step = 0, goal_cell = None, num_envs = 4, seed_base = 0)

   
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

