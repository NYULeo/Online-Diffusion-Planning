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
from torch.utils.data import DataLoader
from Finetuning.utils import cycle
#from Pretrain.Planners.Backbone.utils import get_pretrained_planner
from Finetuning.utils import get_planner, get_normalized_score, get_expert_score, PlannerDataset, get_current_state, reward_processor, check_device
from Pretrain.Dataset import Planner_Processor, get_dataset
from Pretrain.Planners.Backbone.Sampler import sample_reverse_sde, sample_euler_karras, sample_euler_karras2
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv 
import pickle
import random
import gymnasium as gym
import gymnasium_robotics
from Pretrain.Dataset import get_dataset
from gymnasium.wrappers import TimeLimit
from typing import Optional, List
#from utils import get_normalized_score, rollout_parallel3, get_current_state, get_trajs, spare_reward_prcocessor, compute_threshold_log_prob_mog, compute_threshold_mahalanobis_mog
from dataclasses import dataclass
import time
from typing import List
from Finetuning.traj_reward2 import TotalReward_Critic, RewardConfig, TotalReward
class Selector():
    def __init__(self, env_name, specific_env, RConfig: RewardConfig, reward_checkpoint: int, kernel_checkpoint: Optional[int] = None, critic_checkpoint: Optional[int] = None):
         self.env_name = env_name
         self.specific_env = specific_env
         self.RConfig = RConfig
         self.reward_checkpoint = reward_checkpoint
         self.kernel_checkpoint = kernel_checkpoint
         self.critic_checkpoint = critic_checkpoint
         self.device = check_device()
         self.lam = 0.05
         if(critic_checkpoint is not None):
            self.model = TotalReward_Critic(self.device, RConfig, env_name, specific_env, self.reward_checkpoint, self.kernel_checkpoint, self.critic_checkpoint)
         else:
            self.model = TotalReward(self.device, RConfig, env_name, specific_env, self.reward_checkpoint, self.kernel_checkpoint)
         self.model.eval()
    
    def select_plan(self, plans: List[np.ndarray]) -> np.ndarray:
         rewards = []
         with torch.no_grad():
            for plan in plans:
             plan_tensor = torch.from_numpy(plan).float().to(self.device) 
             reward = self.model.predict(plan_tensor, self.lam)
             rewards.append(reward.item())
         return plans[rewards.index(max(rewards))].copy()

def check(env):
    print("Reward type:", getattr(env, 'reward_type', 'Not found'))
    print("Goal distance threshold:")

    # Check the actual reward function   
    if hasattr(env, 'compute_reward'):
    # You can test it
        dummy_achieved = np.array([0.0, 0.0])
        dummy_desired = np.array([0.0, 0.0])
        reward, info = env.compute_reward(dummy_achieved, dummy_desired, None)
        print("Reward when distance=0:", reward)

    # Or manually compute
    pos = env.get_pos()           # current ball position
    goal = env.get_target()       # current goal
    dist = np.linalg.norm(pos - goal)
    print(f"Current distance to goal: {dist:.4f}")
    print(f"Reward will be +1 if distance <= 0.5 → Currently: {dist <= 0.5}")

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

def get_normalized_score(score, min_score,  max_score):
    return (100 * ((score - min_score) / (max_score - min_score)))

@dataclass
class Kernel_Config:
    ensemble_size: int = 10
    num_hidden_layers: int = 2
    hidden_dim: int = 256
    type_kernel: str = 'robust' or 'mog'
    kernel_num_modes: Optional[int] = 8
    kernel_noise_floor: Optional[float] = 1e-4

def feasibility_check(generated_state, new_state):
    return np.linalg.norm(generated_state - new_state)

def get_success_trajs(trajs):
    success_trajs = []
    for traj in trajs:
        if(traj['rewards'][-1] == 1):
            success_trajs.append(traj)
    return success_trajs

def render(dataset_name, specific_dataset, traj, goal_cell, start_cell):
     env, _, _ = get_env(dataset_name, specific_dataset, render_mode = 'rgb_array')
     #env = gym.make("antmaze-medium-v0") 
     obs0 = traj["observations"][0]

     env.reset(seed=0, options = {'goal_cell': goal_cell, 'reset_cell': start_cell})  # optional fixed seed for determinism

    
     frames = []
     rewards = []
     for i in range(len(traj['actions'])):
          action = traj['actions'][i]
            #action = np.clip(action, -1.0, 1.0)
          _, reward, terminated, truncated, _ = env.step(action)
          rewards.append(reward)
          frames.append(env.render())
          if terminated or truncated:
               break
     #print(rewards)
     #print(len(frames))
     media.write_video("demo2.mp4", frames, fps=50)
     env.close()

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
            """
            if dataset_name == 'pointmaze':
                s_critic = s[:2]
            else:
                s_critic = s
            """
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
            """
            if dataset_name == 'pointmaze':
                s_final_critic = s_final[:2]
            else:
                s_final_critic = s_final
            """
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

def save_trajs(trajs, env_name, specific_env, step):
    os.makedirs(f'./Finetuning/Rollouts/{env_name}/{specific_env}/', exist_ok=True)
    save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/Generated_trajs_Info_{str(step)}.pkl'
    with open(save_path, 'wb') as f:
         pickle.dump(trajs, f)
    print(f"trajectories saved")

def save_success_trajs_for_reward(trajs, env_name, specific_env, task_id, step):
    save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{task_id}/trajs_task{task_id}_success_{step}.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajs, f)
    print("trajectories saved")

def load_success_trajs(env_name, specific_env, task_id, step):
    save_path = f'./Finetuning/Rollouts/{env_name}/{specific_env}/task_{task_id}/trajs_task{task_id}_success_{step}.pkl'
    with open(save_path, 'rb') as f:
        trajs = pickle.load(f)
    return trajs

def rollout(env_name, specific_env, horizon, steps_T, num_karras, eta, episode_length, checkpoint_steps, render = False, goal_cell: Optional[np.ndarray] = None, start_cell: Optional[np.ndarray] = None, task_id: Optional[int] = None, base_seed: int = 0, continual_rollout = False, chunk_size = 5, device = None, selector: Optional[Selector] = None):
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     #print(f"Horizon: {horizon}, step_T: {steps_T}, num_karras: {num_karras}, eta: {eta}, Checkpoint_steps; {checkpoint_steps}, episode_length: {episode_length}")
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     #device = check_device()
     #device = "cuda" if torch.cuda.is_available() else "cpu"
     #print(f"Using device {device}")
     import minari
     #env.reset(seed=1)  # Important: pass seed to env.reset
     env, d_s, d_a = get_env(env_name, specific_env, render_mode = 'rgb_array', task_id = task_id, episode_length = None)
     #env, d_s, d_a = get_env(env_name, specific_env, render_mode = 'rgb_array', episode_length = episode_length)
     #np.random.seed(base_seed)
    
    # 2. Reset environment with both seed and task_id
     #env.reset(seed=base_seed)   # Important first reset
    
    # Create environment factory function
     state_dict = get_planner(env_name, specific_env, checkpoint_steps, task_id)
     if( env_name == 'kitchen'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif (env_name == 'pointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif(env_name == 'antmaze'):
           model = DiT1d(in_dim = (d_s), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif(env_name == 'cube'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif(env_name == 'ogpointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     else:
          raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()

     #get Processor
     planner_processor = Planner_Processor(env_name, specific_env, task_id)
     
     
     #reset
     if(env_name == 'cube'):
         s0, info = env.reset(seed = base_seed, options = dict( task_id=task_id))
         #s0, info = env.reset(seed = base_seed)
        #s0, info = env.reset()
     elif(env_name == 'ogpointmaze'):
         s0, info = env.reset(seed = base_seed, options = dict( task_id=task_id))

     elif(goal_cell is not None and start_cell is not None):
         s0 = env.reset(seed = base_seed, options = {"goal_cell": goal_cell, "reset_cell": start_cell})
        #s0, info = env.reset( options = {"goal_cell": goal_cell, "reset_cell": start_cell})
     elif(goal_cell is not None):
         s0 = env.reset(seed = base_seed, options = {"goal_cell": goal_cell})
     else:
         s0 = env.reset(seed = base_seed)
        #s0, info = env.reset()
     
     
     current_state = get_current_state(s0[0], env_name)
     frames = []
     observations = []
     actions = []
     rewards = []
     Temp_acts = []
     Temp_states = []
     generated_state = None
     violation_scores = []
     number_of_plans = 0
     for i in range(episode_length):
           if(continual_rollout):
                if(len(Temp_acts) == 0):
                     current_state_norm = planner_processor.preprocess(current_state)
                     if(selector is None):
                         x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
                     else:
                         Plans = []
                         for j in range(30):
                              Plans.append(sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device))
                         x = selector.select_plan(Plans)
                     for k in range(min(chunk_size, len(x))):
                         Temp_acts.append(x[k, d_s:(d_s+d_a)].copy())
                     for k in range(1, min(chunk_size, len(x))):
                         Temp_states.append(x[k, :d_s].copy())
                     number_of_plans += 1
                
                action = Temp_acts[0]
                Temp_acts = Temp_acts[1:]
                if(len(Temp_states)> 0):
                    generated_state = Temp_states[0]
                    Temp_states = Temp_states[1:]
                else:
                    generated_state = None

                obs, reward, terminated, truncated, info = env.step(action)
                if(render):
                      frames.append(env.render())
           else:
                current_state_norm = planner_processor.preprocess(current_state)
                #x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta,  device = device)
                if(selector is None):
                    x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
                else:
                    Plans = []
                    for j in range(30):
                        Plans.append(sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device))
                    x = selector.select_plan(Plans)
                action = x[0, d_s:(d_s+d_a)].copy()
                generated_state = x[1, :d_s].copy()
                obs, reward, terminated, truncated, info = env.step(action)
                if(render):
                      frames.append(env.render())
           
           
           current_state = get_current_state(obs.copy(), env_name)
           if(generated_state is not None):
                violation_scores.append(feasibility_check(generated_state, current_state.copy()))
           observations.append(current_state.copy())
           actions.append(action.copy())
           rewards.append(reward)
           #current_state = obs['observation'].copy()
           #print(f"Episode {i} reward: {reward}")
           if(terminated or truncated):
                #print(f"Episode {i} terminated or truncated")
                break
     
     env.close()

     
     """
     if(len(violation_scores) > 0):
         print(np.mean(violation_scores))
         print(np.var(violation_scores))
     """
     #print(f"total steps: {len(observations)}")
     #print(f"number of plans: {number_of_plans}")
     rewards = reward_processor(rewards, env_name)
     traj = {'observations': np.asarray(observations), 'actions': np.asarray(actions), 'rewards': np.asarray(rewards)}
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
     #print(sum(traj['rewards']))
     #return traj
     
     #return rewards[-1], len(observations)
     return sum(rewards), len(observations)
     #print(get_normalized_score([traj]))
 
def load_kernel(env_name, specific_env, checkpoint_steps, kernel_config: Kernel_Config, device: str):
    from Pretrain.Transition_Kernel.Kernel_Backbone import MoGTransitionKernel
    from Finetuning.utils import get_kernel, get_kernel_stats
    kernel_state_dicts, obs_dim, act_dim = get_kernel(env_name, specific_env, checkpoint_steps)
    kernels = []
    kernel_stats = get_kernel_stats(env_name, specific_env, checkpoint_steps)
    Model = MoGTransitionKernel
    for sd in kernel_state_dicts:
            kernel_net = Model(
                obs_dim, act_dim, kernel_config.kernel_num_modes, kernel_config.num_hidden_layers, kernel_config.hidden_dim, noise_floor = kernel_config.kernel_noise_floor
            ).to(device)
            kernel_net.load_state_dict(sd)
            kernel_net.eval()
            kernels.append(kernel_net)
    return kernels, kernel_stats, obs_dim, act_dim

def compute_log_prob(kernels, kernel_stats, x, obs_dim, act_dim, type: str = 'log_density', device: str = 'cuda'):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from Pretrain.Transition_Kernel.Kernel_Backbone import  compute_log_density_mog, compute_total_mahalanobis_score_mog
    values = []
    for i in range(1, len(x)-1):
        obs = torch.tensor(kernel_stats.norm_obs(x[i, :obs_dim].copy()), dtype = torch.float32).unsqueeze(0).to(device)
        act = torch.tensor(x[i, obs_dim:(obs_dim+act_dim)].copy(), dtype = torch.float32).unsqueeze(0).to(device)
        s_next = torch.tensor(kernel_stats.norm_obs(x[i+1, :obs_dim].copy()), dtype = torch.float32).unsqueeze(0).to(device)
        if(type == 'log_density'):
            value = compute_log_density_mog(kernels, obs, act, s_next).item()
        else:
            value = compute_total_mahalanobis_score_mog(kernels, obs, act, s_next).item()
        values.append(value)
    return np.mean(values)

def Test_Kernel_on_Generated_Trajs(env_name, specific_env, horizon, kernel_config: Kernel_Config,  steps_T, num_karras, eta, time, planner_checkpoint, kernel_checkpoint, task_id: Optional[int] = None):
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     

     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     device = check_device()
     print(f"Using device {device}")
     
     _, d_s, d_a = get_env(env_name, specific_env, render_mode = 'rgb_array')
    
    # Create environment factory function
     state_dict = get_planner(env_name, specific_env, planner_checkpoint)
     if( env_name == 'kitchen'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif (env_name == 'pointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif(env_name == 'antmaze'):
           model = DiT1d(in_dim = (d_s), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     elif(env_name == 'cube'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier").to(device)
     else:
          raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()
     

     #get Processor
     planner_processor = Planner_Processor(env_name, specific_env)

     dataset = get_dataset(env_name, specific_env, task_id)
     trajs = dataset.get_trajectories()
     planner_dataset = PlannerDataset(trajs, horizon, env_name, specific_env)
     dataloader = cycle(DataLoader(planner_dataset, batch_size = 1, shuffle = False))
     kernels, kernel_stats, obs_dim, act_dim = load_kernel(env_name, specific_env, kernel_checkpoint, kernel_config, device)
     mahalanobis_scores = []
     log_density_scores = []
     for i in range(time):
            norm_state = next(dataloader)
            norm_state = norm_state.squeeze(0).numpy()
            x = sample_euler_karras(norm_state, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
            log_density_score = compute_log_prob(kernels, kernel_stats, x, obs_dim, act_dim, type = 'log_density', device = device)
            mahalanobis_score = compute_log_prob(kernels, kernel_stats, x, obs_dim, act_dim, type = 'mahalanobis', device = device)
            mahalanobis_scores.append(mahalanobis_score)
            log_density_scores.append(log_density_score)
           
     print(f"Mean of Mahalanobis scores: {np.mean(mahalanobis_scores):.4f}")
     print(f"Max of Mahalanobis scores: {np.max(mahalanobis_scores):.4f}")
     print(f"Min of Mahalanobis scores: {np.min(mahalanobis_scores):.4f}")
     print(f"STD of Mahalanobis scores: {np.std(mahalanobis_scores):.4f}")
     print(f"quantile 0.95 of Mahalanobis scores: {np.quantile(mahalanobis_scores, 0.95):.4f}")

     print("--------------------------------------------------------------------------------------------------")
     print(f"Mean of log_density scores: {np.mean(log_density_scores):.4f}")
     print(f"Max of log_density scores: {np.max(log_density_scores):.4f}")
     print(f"Min of log_density scores: {np.min(log_density_scores):.4f}")
     print(f"STD of log_density scores: {np.std(log_density_scores):.4f}")
     print(f"quantile 0.05 of Mahalanobis scores: {np.quantile(log_density_scores, 0.05):.4f}")
    
     
   
     #return len(traj['rewards'])
     #print(get_normalized_score([traj]))


# ---- 4) Example usage (fill ScoreWrapper first) ----
if __name__ == "__main__":
    horizon = 32
    env_name = 'cube'
    specific_train_dataset = 'single-play'
    task_id = 4
    checkpoint = 21
    total_reward = 0.0
    device = check_device()
    print(f"Using device {device}")
    RConfig = RewardConfig(
               beta = 1.0, 
               #max_mahalanobis_score = 3.5,
               min_log_prob = 5.0,
               #constraint_adapt = False,
               critic_gamma = 1.0,
               type_kernel = 'mog',
               kernel_num_modes = 10,
               kernel_noise_floor = 5e-4,
               num_hidden_layers_kernel = 2,
               hidden_dim_kernel = 256,
               num_hidden_layers_reward = 1,
               hidden_dim_reward = 32,
               num_hidden_layers_critic = 3,
               hidden_dim_critic = 256,
               explore = False)
    #selector = Selector(env_name, specific_train_dataset, RConfig, reward_checkpoint = 60, kernel_checkpoint = 60, critic_checkpoint = None)
    #chunk_size = [31, 25, 20, 19, 18, 13, 12, 11, 10, 15, 7, 6, 8, 5, 16, 4, 9, 14, 17]
    chunk_size = [31, 25, 20, 19, 18, 13, 12, 11, 10, 15, 7, 6, 8, 5, 16, 4, 9, 14, 17, 21, 22, 23, 24, 26, 27, 28, 29, 30]
    set_seed(8)
    while(checkpoint < 24):
       print(f"Running checkpoing: {checkpoint}")
       total_return = 0.0
       for j in range(1, 51):
          return_value = 0.0
          chunk_size_index = 0
          while((return_value != 1.0) and (chunk_size_index < len(chunk_size))):
                return_value, _ = rollout(
                  env_name, 
                  specific_train_dataset, 
                  horizon, 
                  steps_T = 10, 
                  num_karras = 1, 
                  eta = 0.0, 
                  episode_length = 3000, 
                  checkpoint_steps = checkpoint, 
                  render = False,  
                  base_seed = j, 
                  #goal_cell = np.array([6, 1], dtype = int), 
                  task_id = task_id,
                  continual_rollout = True,
                  chunk_size = chunk_size[chunk_size_index],
                  device = device)
                chunk_size_index += 1
          print(return_value)
          total_return += return_value
       print(f"Checkpoint: {checkpoint} Success Rate: {total_return / 50 :.4f}")
       checkpoint += 3
    exit()

    