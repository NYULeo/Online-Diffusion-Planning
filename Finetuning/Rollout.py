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
from utils import get_pretrained_planner
from Pretrain.Dataset import Planner_Processor
from Pretrain.Planners.Backbone.Sampler import sample_reverse_sde, sample_euler_karras, sample_euler_karras2
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv 
import pickle
import random

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

def rollout(env_name, specific_env, horizon, steps_T, num_karras, eta, episode_length, checkpoint_steps, render = False):
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, Checpoint_steps; {checkpoint_steps}")
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     device = "cuda" if torch.cuda.is_available() else "cpu"
     print(f"Using device {device}")
     
     #get environment
     
     if(render):
         env, d_s, d_a = get_env(env_name, specific_env, 'rgb_array')
     else:
         env, d_s, d_a = get_env(env_name, specific_env, None)
     #env = gym.make('PointMaze_Medium-v3', render_mode = 'rgb_array')

     #get Planner
     state_dict = get_pretrained_planner(env_name, specific_env, checkpoint_steps)
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
     s0 = env.reset(seed=1)
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
     

def rollout_parallel(env_name, specific_env, horizon = 32, steps_T = 50, num_karras = 10, eta = 0.8, episode_length = 4000, critic = False, checkpoint_steps = 1000000, num_envs=8):
     
     #print(f"Horizon: {horizon}, step_T: {steps_T}, eta: {eta}, critic: {critic}, Checkpoint_steps: {checkpoint_steps}")
     #print(f"Running {num_envs} environments in parallel")
     
     device = "cuda" if torch.cuda.is_available() else "cpu"
     #print(f"Using device {device}")
     
     
     # Create environment factory function
     def make_env():
         env, _, _ = get_env(env_name, specific_env)
         return env
     
     # Create vectorized environment
     vec_env = AsyncVectorEnv([make_env for _ in range(num_envs)])
     
     # Get dimensions from single env
     _, d_s, d_a = get_env(env_name, specific_env)
     
     # Get Planner
     state_dict = get_pretrained_planner(env_name, specific_env, checkpoint_steps)
     if env_name == 'kitchen':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     elif env_name == 'pointmaze':
         model = DiT1d(in_dim=(d_s + d_a), emb_dim=128, d_model=256, n_heads=256//64, depth=2, timestep_emb_type="fourier").to(device)
     else:
         raise ValueError(f"Invalid Environment: {env_name}")
     model.load_state_dict(state_dict)
     model.eval()
     
     # Get Processor
     planner_processor = Planner_Processor(env_name, specific_env)
     
     # Reset all environments
     s0_vec = vec_env.reset(seed=1)
     current_states = s0_vec[0]['observation']  # Shape: (num_envs, d_s)
     
     # Store trajectories for each environment
     all_rewards = [0.0 for _ in range(num_envs)]
     done_envs = [False for _ in range(num_envs)]
     observations = [[] for _ in range(num_envs)]
     acts = [[] for _ in range(num_envs)]
     rewards = [[] for _ in range(num_envs)]
     
     for i in range(episode_length):
         actions = np.zeros((num_envs, d_a))
         
         # Generate actions for each environment
         for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
                 
             current_state = current_states[env_idx]
             current_state_norm = planner_processor.preprocess(current_state)
             
             x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
             #x = sample_reverse_sde(current_state_norm, model, d_s, d_a, horizon, steps_T, eta, device)
             action = x[0, d_s:(d_s+d_a)].copy()
             
             actions[env_idx] = action
         
         # Step all environments at once
         obs_vec, rewards_vec, terminated_vec, truncated_vec, info_vec = vec_env.step(actions)
         
         # Update trajectories
         for env_idx in range(num_envs):
             if done_envs[env_idx]:
                 continue
             
             observations[env_idx].append(obs_vec['observation'][env_idx].copy())
             acts[env_idx].append(actions[env_idx].copy())
             rewards[env_idx].append(rewards_vec[env_idx])
             all_rewards[env_idx] += rewards_vec[env_idx]
             
             current_states[env_idx] = obs_vec['observation'][env_idx].copy()
             
             if terminated_vec[env_idx] or truncated_vec[env_idx]:
                 done_envs[env_idx] = True
                 #print(f"Env {env_idx} finished at step {i}, total reward: {all_rewards[env_idx]:.4f}")
         
        
         # Check if all environments are done
         if all(done_envs):
             #print("All environments completed!")
             break
         
     
     vec_env.close()
     
     # Find the trajectory with the maximum reward
     trajs = [[] for _ in range(num_envs)]
     for env_idx in range(num_envs):
         trajs[env_idx] = {
             'observations': np.asarray(observations[env_idx].copy()),
             'actions': np.asarray(acts[env_idx].copy()),
             'rewards': np.asarray(rewards[env_idx].copy())
         }
     #best_idx = np.argmax(all_rewards)
     #best_reward = all_rewards[best_idx]
     #best_trajectory = trajs[best_idx]

     # Save the best trajectory in the same format as single rollout
     """
     trajs_info = {
         'trajs': trajs,
         'env_name': env_name,
         'specific_env': specific_env,
         'all_rewards': all_rewards
     }
     save_trajs(trajs_info, env_name, specific_env)
     """
     
    
     





# ---- 4) Example usage (fill ScoreWrapper first) ----
if __name__ == "__main__":
    set_seed(1)
    horizon = 32
    env_name = 'pointmaze'
    specific_train_dataset = 'medium'
    rollout(env_name, specific_train_dataset, horizon, steps_T = 50, num_karras = 10, eta = 1.0, episode_length  = 3000, checkpoint_steps = 400, render = True)
    #rollout_parallel(env_name, specific_train_dataset, horizon, steps_T = 200, eta = 0.8, episode_length  = 10000, critic = False, checkpoint_steps = 1500, num_envs = 50)

