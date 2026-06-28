'''Rollout / planning-time sampling utilities (JAX/Flax port, FQL-style).'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
# Disable XLA Triton-GEMM autotuning BEFORE importing jax (running `python Rollout.py` directly does not
# go through run_cube_pipeline.py, so without this the dot_search_space autotuning storm makes eval crawl).
if 'XLA_FLAGS' not in os.environ and os.environ.get('ODP_AUTOTUNE', '0') != '1':
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_autotune_level=0'
        ' --xla_gpu_enable_triton_gemm=false'
        ' --xla_gpu_exhaustive_tiling_search=false'
        ' --xla_gpu_cublas_fallback=true'
    )
import jax
import jax.numpy as jnp
import numpy as np
import mediapy as media
from Pretrain.Dataset import get_env
from Pretrain.Planners.Backbone.Dit import DiT1d
from Finetuning.utils import cycle
#from Pretrain.Planners.Backbone.utils import get_pretrained_planner
from Finetuning.utils import get_planner, get_normalized_score, get_expert_score, PlannerDataset, get_current_state, reward_processor, check_device
from Pretrain.Dataset import Planner_Processor, get_dataset
from Pretrain.Planners.Backbone.Sampler import sample_reverse_sde, sample_euler_karras, sample_euler_karras2
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import pickle
import random
import gymnasium as gym
try:
    import gymnasium_robotics  # registers FrankaKitchen; only needed for the kitchen env (not cube)
except ImportError:
    gymnasium_robotics = None
from Pretrain.Dataset import get_dataset
from gymnasium.wrappers import TimeLimit
from typing import Optional
#from utils import get_normalized_score, rollout_parallel3, get_current_state, get_trajs, spare_reward_prcocessor, compute_threshold_log_prob_mog, compute_threshold_mahalanobis_mog
from dataclasses import dataclass
import time
from typing import List
from Finetuning.traj_reward import TotalReward_Critic, RewardConfig, TotalReward
from Pretrain.Planners.Backbone.Sampler import karras_beta_schedule, cosine_beta, clip_actions

from flax_utils import TrainState


def create_initial(current_state: np.ndarray, plan_suffix: np.ndarray, d_s: int, d_a: int, horizon: int, device: str) -> np.ndarray:
    initial = jnp.zeros((1, horizon, d_s + d_a))
    initial = initial.at[:, 0, :d_s].set(current_state)
    initial = initial.at[:, 0, d_s:(d_s + d_a)].set(plan_suffix[0, d_s:(d_s + d_a)])
    for i in range(horizon):
        if (i < len(plan_suffix)):
            initial = initial.at[:, i].set(plan_suffix[i])
        else:
            initial = initial.at[:, i].set(initial[:, i - 1])
    return initial

def sample_euler_karras_replan(
    s0: np.ndarray,
    score_model,
    d_s: int,
    d_a: int,
    horizon: int,
    num_steps: int = 50,
    num_karras: int = 5,
    eta: float = 1.0,
    plan_suffix: np.ndarray = None,
    device: Optional[str] = None,
    *,
    rng=None,
) -> np.ndarray:
    # API-CHANGE: added keyword-only `rng=` (Euler-Karras noise is stochastic when eta > 0).
    s0_t = jnp.asarray(s0, dtype=jnp.float32)
    if s0_t.shape[0] != d_s:
        raise ValueError(f"s0 should have shape ({d_s},), but got {s0_t.shape}")

    dim = d_s + d_a

    # Karras β(t) + σ(t)
    t_grid, beta_1, sigma_grid = karras_beta_schedule(num_steps, device=device)
    #t_grid, beta_1, _, sigma_grid =  karras_cosine_interpolated_beta(num_steps, device=device)

    beta_2 = cosine_beta(t_grid, s=0.008)

    # Initialize x_T
    x = create_initial(s0, plan_suffix, d_s, d_a, horizon, device) * sigma_grid[0]
    #x = torch.randn(1, horizon, dim, device=device) * sigma_grid[0]
    #x2 = torch.randn(1, horizon, dim, device=device)

    # Conditioning
    mask = jnp.zeros((1, horizon, dim))
    mask = mask.at[:, 0, :d_s].set(1.0)
    y = jnp.zeros_like(x)
    y = y.at[:, 0, :d_s].set(s0_t[None])
    x = mask * y + (1 - mask) * x

    for i in range(num_steps):
        t_now = t_grid[i]
        t_next = t_grid[i + 1] if i < num_steps - 1 else 0.0
        dt = float(t_next - t_now)
        if (i < num_karras):
            beta_now = float(beta_1[i])
        else:
            beta_now = float(beta_2[i])

        # Drift
        drift = -0.5 * beta_now * x

        # Score
        score = score_model(x, t_now[None])

        # Euler step
        if eta > 0:
            rng, noise_key = jax.random.split(rng)
            noise = jax.random.normal(noise_key, x.shape)
            noise_scale = eta * jnp.sqrt(beta_now * (-dt))
            x = x + ((drift - beta_now * score) * dt + noise_scale * noise)
        else:
            x = x + (drift - beta_now * score) * dt

        # Conditioning
        x = mask * y + (1 - mask) * x
        x = clip_actions(x, d_s)

    return np.asarray(x.squeeze(0))

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

    def select_plan(self, plans: List[np.ndarray]) -> np.ndarray:
         rewards = []
         for plan in plans:
             plan_tensor = jnp.asarray(plan, dtype=jnp.float32)
             reward = self.model.predict(plan_tensor, self.lam)
             rewards.append(float(reward))
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
        device = None  # API note: JAX places arrays automatically; `device` kept for signature compat.

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
    # TODO(checkpoint-bridge): map the torch reward state_dict into a flax param tree + TrainState.
    reward_net = SimpleReward(obs_dim, act_dim)
    reward_params = reward_state_dict
    reward_state = TrainState.create(reward_net, reward_params)
    reward_stats = get_reward_stats(dataset_name, specific_dataset, reward_checkpoint)

    # Load kernel models and stats
    kernel_state_dicts, _, _ = get_kernel(dataset_name, specific_dataset, kernel_checkpoint)
    kernels = []
    for kernel_state_dict in kernel_state_dicts:
        # TODO(checkpoint-bridge): map the torch kernel state_dict into a flax param tree + TrainState.
        kernel_net = RobustTransitionKernel(obs_dim, act_dim)
        kernel_state = TrainState.create(kernel_net, kernel_state_dict)
        kernels.append(kernel_state)
    kernel_stats = get_kernel_stats(dataset_name, specific_dataset, kernel_checkpoint)

    # Load critic model and stats
    critic_state_dict, critic_obs_dim = get_critic_model(dataset_name, specific_dataset, critic_checkpoint)
    # TODO(checkpoint-bridge): map the torch critic state_dict into a flax param tree + TrainState.
    critic_net = Critic(critic_obs_dim)
    critic_state = TrainState.create(critic_net, critic_state_dict)
    critic_stats = get_critic_stats(dataset_name, specific_dataset, critic_checkpoint)

    observations = traj['observations']
    actions = traj['actions']

    # Calculate average log probability, average reward, and average critic value
    total_log_prob = 0.0
    total_reward = 0.0
    total_critic = 0.0
    num_transitions = len(actions)
    num_states = len(observations)
    
    for t in range(num_transitions):
        # Get state, action, and next state
        s = observations[t]
        a = actions[t]

        # Compute reward
        s_norm_reward = reward_stats.norm_obs(s)
        s_tensor = jnp.asarray(s_norm_reward, dtype=jnp.float32)[None]
        a_tensor = jnp.asarray(a, dtype=jnp.float32)[None]
        r = reward_state(s_tensor, a_tensor)
        total_reward += float(r)

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
        s_critic_tensor = jnp.asarray(s_norm_critic, dtype=jnp.float32)[None]
        v = critic_state(s_critic_tensor)
        total_critic += float(v)

        # Skip if we don't have next state for log prob calculation
        if t >= len(observations) - 1:
            continue

        s_next = observations[t + 1]

        # Compute log probability using kernel ensemble
        s_norm_kernel = kernel_stats.norm_obs(s)
        s_next_norm_kernel = kernel_stats.norm_obs(s_next)

        s_tensor = jnp.asarray(s_norm_kernel, dtype=jnp.float32)[None]
        a_tensor = jnp.asarray(a, dtype=jnp.float32)[None]
        s_next_tensor = jnp.asarray(s_next_norm_kernel, dtype=jnp.float32)[None]

        # Average log prob across ensemble
        ensemble_log_probs = []
        for kernel in kernels:
            mu, log_std = kernel(s_tensor, a_tensor)
            lp = kernel(s_next_tensor, mu, log_std, method='log_prob')
            ensemble_log_probs.append(float(lp))

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
        s_final_critic_tensor = jnp.asarray(s_final_norm_critic, dtype=jnp.float32)[None]
        v_final = critic_state(s_final_critic_tensor)
        total_critic += float(v_final)
    
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
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    # JAX has no global RNG: return a key for callers to thread.
    return jax.random.PRNGKey(seed)

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

def rollout(env_name,
            specific_env,
            horizon,
            steps_T, 
            num_karras, eta, 
            episode_length, 
            checkpoint_steps, 
            render = False, 
            goal_cell: Optional[np.ndarray] = None, 
            start_cell: Optional[np.ndarray] = None,
            task_id: Optional[int] = None,
            base_seed: int = None, 
            continual_rollout = False,
            chunk_size = 5,
            device = None,
            selector: Optional[Selector] = None,
            *,
            rng=None):
     # API-CHANGE: added keyword-only `rng=` (planner sampling is stochastic). Defaults from base_seed.
     if rng is None:
          rng = jax.random.PRNGKey(base_seed if base_seed is not None else 0)
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     #print(f"Horizon: {horizon}, step_T: {steps_T}, num_karras: {num_karras}, eta: {eta}, Checkpoint_steps; {checkpoint_steps}, episode_length: {episode_length}")
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     #device = check_device()
     #device = "cuda" if torch.cuda.is_available() else "cpu"
     #print(f"Using device {device}")

     #env.reset(seed=1)  # Important: pass seed to env.reset
     env, d_s, d_a = get_env(env_name, specific_env, render_mode = 'rgb_array', task_id = task_id, episode_length = None)
     #env, d_s, d_a = get_env(env_name, specific_env, render_mode = 'rgb_array', episode_length = episode_length)
     #np.random.seed(base_seed)
    
    # 2. Reset environment with both seed and task_id
     #env.reset(seed=base_seed)   # Important first reset
    
    # Create environment factory function
     state_dict = get_planner(env_name, specific_env, checkpoint_steps, task_id)
     #state_dict = get_planner(env_name, specific_env, checkpoint_steps)
     if( env_name == 'kitchen'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     elif (env_name == 'pointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     elif(env_name == 'antmaze'):
           model = DiT1d(in_dim = (d_s), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     elif(env_name == 'cube'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     elif(env_name == 'ogpointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     else:
          raise ValueError(f"Invalid Environment: {env_name}")
     # TODO(checkpoint-bridge): map the torch planner state_dict into a flax param tree.
     model = TrainState.create(model, state_dict)

     #get Processor
     planner_processor = Planner_Processor(env_name, specific_env, task_id)
     #planner_processor = Planner_Processor(env_name, specific_env)
     
     
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
                         rng, plan_key = jax.random.split(rng)
                         x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=plan_key)
                     else:
                         Plans = []
                         for j in range(30):
                              rng, plan_key = jax.random.split(rng)
                              Plans.append(sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=plan_key))
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
                    rng, plan_key = jax.random.split(rng)
                    x = sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=plan_key)
                else:
                    Plans = []
                    for j in range(30):
                        rng, plan_key = jax.random.split(rng)
                        Plans.append(sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=plan_key))
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
     
     
     #print(rewards)
     rewards = reward_processor(rewards, env_name)
     #print(rewards)
     #print(rewards)
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
            )
            # TODO(checkpoint-bridge): map the torch kernel state_dict into a flax param tree.
            # §11: compute_*_mog consumers iterate `for model_def, params in kernels`, so store tuples.
            kernels.append((kernel_net, sd))
    return kernels, kernel_stats, obs_dim, act_dim

def compute_log_prob(kernels, kernel_stats, x, obs_dim, act_dim, type: str = 'log_density', device: str = 'cuda'):
    from Pretrain.Transition_Kernel.Kernel_Backbone import  compute_log_density_mog, compute_total_mahalanobis_score_mog
    values = []
    for i in range(1, len(x)-1):
        obs = jnp.asarray(kernel_stats.norm_obs(x[i, :obs_dim].copy()), dtype=jnp.float32)[None]
        act = jnp.asarray(x[i, obs_dim:(obs_dim+act_dim)].copy(), dtype=jnp.float32)[None]
        s_next = jnp.asarray(kernel_stats.norm_obs(x[i+1, :obs_dim].copy()), dtype=jnp.float32)[None]
        if(type == 'log_density'):
            value = float(compute_log_density_mog(kernels, obs, act, s_next)[0])
        else:
            value = float(compute_total_mahalanobis_score_mog(kernels, obs, act, s_next)[0])
        values.append(value)
    return np.mean(values)

def Test_Kernel_on_Generated_Trajs(env_name, specific_env, horizon, kernel_config: Kernel_Config,  steps_T, num_karras, eta, time, planner_checkpoint, kernel_checkpoint, task_id: Optional[int] = None, *, rng=None):
     # API-CHANGE: added keyword-only `rng=` (planner sampling is stochastic).
     if rng is None:
          rng = jax.random.PRNGKey(0)
     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     

     #env = gym.make('FrankaKitchen-v1',  tasks_to_complete = ['microwave', 'kettle', 'light switch', 'slide cabinet'], render_mode = None)  # Use headless mode for servers
     device = check_device()
     print(f"Using device {device}")
     
     _, d_s, d_a = get_env(env_name, specific_env, render_mode = 'rgb_array')
    
    # Create environment factory function
     state_dict = get_planner(env_name, specific_env, planner_checkpoint)
     if( env_name == 'kitchen'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     elif (env_name == 'pointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     elif(env_name == 'antmaze'):
           model = DiT1d(in_dim = (d_s), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     elif(env_name == 'cube'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= 2, timestep_emb_type="fourier")
     else:
          raise ValueError(f"Invalid Environment: {env_name}")
     # TODO(checkpoint-bridge): map the torch planner state_dict into a flax param tree.
     model = TrainState.create(model, state_dict)


     #get Processor
     planner_processor = Planner_Processor(env_name, specific_env)

     dataset = get_dataset(env_name, specific_env, task_id)
     trajs = dataset.get_trajectories()
     planner_dataset = PlannerDataset(trajs, horizon, env_name, specific_env)

     def _dataloader_cycle(ds):
          # Replaces torch DataLoader(batch_size=1, shuffle=False) + cycle: sequential numpy batching.
          while True:
               for idx in range(len(ds)):
                    yield np.asarray(ds[idx])

     dataloader = cycle(_dataloader_cycle(planner_dataset))
     kernels, kernel_stats, obs_dim, act_dim = load_kernel(env_name, specific_env, kernel_checkpoint, kernel_config, device)
     mahalanobis_scores = []
     log_density_scores = []
     for i in range(time):
            norm_state = next(dataloader)
            rng, plan_key = jax.random.split(rng)
            x = sample_euler_karras(norm_state, model, d_s, d_a, horizon, steps_T, num_karras, eta, device, rng=plan_key)
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
    goals = {'task_1': np.array( [ 0.0,       -1.0,        0.199599]), 
         'task_2': np.array([7.50000000e-01, 8.02418254e-18, 1.99598996e-01]),
         'task_3': np.array([-7.50000000e-01,  1.21832368e-19,  1.99598996e-01]),
         'task_4': np.array([0.75,     2.0,       0.199599]),
         'task_5': np.array([ 0.75,     -2.0,        0.199599])}
    
   
    """
    horizon = 32
    env_name = 'pointmaze'
    specific_train_dataset = 'medium'
    set_seed(1)
    
    rollout(env_name, 
            specific_train_dataset, horizon, 
            steps_T = 50, 
            num_karras = 3, 
            eta = 0.8, 
            episode_length = 3000, 
            checkpoint_steps = 0, 
            render = True,  
            base_seed = 1, 
            goal_cell = np.array([6, 1], dtype = int), 
            start_cell = np.array([1, 6], dtype = int), 
            continual_rollout = False,
            chunk_size = 10)
    """
    """
    horizon = 70
    env_name = 'pointmaze'
    specific_train_dataset = 'large'
    set_seed(1)
    
    rollout(env_name, 
            specific_train_dataset, horizon, 
            steps_T = 200, 
            num_karras = 10, 
            eta = 0.8, 
            episode_length = 3000, 
            checkpoint_steps = 0, 
            render = True,  
            base_seed = 1, 
            goal_cell = np.array([7, 10], dtype = int), 
            start_cell = np.array([1, 1], dtype = int), 
            continual_rollout = True,
            chunk_size = 10)
    """
    """
    horizon = 32
    env_name = 'kitchen'
    specific_train_dataset = 'partial'
    set_seed(2)
    
    rollout(env_name, 
            specific_train_dataset, 
            horizon, 
            steps_T = 500, 
            num_karras = 25, 
            eta = 0.8, 
            episode_length = 3000, 
            checkpoint_steps = 40, 
            render = True,  
            base_seed = 1, 
            continual_rollout = True,
            chunk_size = 10)
    """

    #rollout(env_name, specific_train_dataset, horizon, steps_T = 150, num_karras = 8, eta = 0.8, episode_length = 1000, checkpoint_steps = 0, render = True, base_seed = 0, continual_rollout = True, chunk_size = 3)
    #traj = model_rollout(env_name, specific_train_dataset, horizon, steps_T = 50, num_karras = 3, eta = 0.8, checkpoint_steps = 210, train_goal = np.array([-2.5, -2.5], dtype = np.float32), rollout_goal = np.array([6, 1], dtype = int), start_cell = np.array([4, 4], dtype = int))
    #print(traj['observations'])
    #render(env_name, specific_train_dataset, traj, goal_cell = np.array([6, 1], dtype = int), start_cell = np.array([4, 4], dtype = int))
    """
    average = 0.0
    for i in range(1, 10):
        set_seed(i)
        score = rollout(env_name, specific_train_dataset, horizon, steps_T = 50, num_karras = 3, eta = 0.8, episode_length = 500, checkpoint_steps = 60, render = True,  goal_cell = np.array([6, 1], dtype = int), start_cell = np.array([6, 5], dtype = int), base_seed = 0, continual_rollout = False)
        average += score
    average = average / 10
    print(average)
    """
    #score = rollout(env_name, specific_train_dataset, horizon, steps_T = 50, num_karras = 3, eta = 0.8, episode_length = 500, checkpoint_steps = 50, render = True,  goal_cell = np.array([6, 1], dtype = int), start_cell = np.array([1, 5], dtype = int), base_seed = 0, continual_rollout = False)
   
    #min_score = 13.13
    #max_score = 277.39
    """
    horizon = 32
    env_name = 'pointmaze'
    specific_train_dataset = 'medium'
    data = get_dataset(env_name, specific_train_dataset)
    min_score = data.get_ref_min_score()
    max_score = data.get_ref_max_score()
    RConfig = RewardConfig(
               beta = 1.0, 
               #max_mahalanobis_score = 3.5,
               min_log_prob = 5.0,
               #constraint_adapt = False,
               critic_gamma = 1.0,
               num_hidden_layers_kernel = 2,
               hidden_dim_kernel = 256,
               num_hidden_layers_reward = 1,
               hidden_dim_reward = 32,
               num_hidden_layers_critic = 3,
               hidden_dim_critic = 256,
               explore = False)
    #selector = Selector(env_name, specific_train_dataset, RConfig, reward_checkpoint = 60, kernel_checkpoint = 60, critic_checkpoint = None)
    device = check_device()
    set_seed(1)
    total_return = 0
    i = 26
    total_steps = 0
    while(True):
       return_value, steps = rollout(env_name, 
            specific_train_dataset, 
            horizon, 
            steps_T = 50, 
            num_karras = 3, 
            eta = 0.8, 
            episode_length = 3000, 
            checkpoint_steps = 80, 
            render = True,  
            base_seed = i, 
            goal_cell = np.array([6, 1], dtype = int), 
            start_cell = np.array([6, 5], dtype = int),
            continual_rollout = True,
            chunk_size = 31,
            device = device,
            selector = None)
       total_steps += steps
       print(total_steps)
       print(return_value)
       exit()
       if(total_steps > 10000):
           break
       else:
           total_return += return_value
           i += 1
    print(total_return)
    print(get_normalized_score(total_return, min_score, max_score))
    
    #print(f"Elapsed: {elapsed:.3f}s")

    """

    
  
    import ogbench
    horizon = 32  # pyright: ignore[reportUnreachable]
    env_name = 'cube'
    specific_train_dataset = 'single-play'   # match the run you trained (was 'double-play')
    task_id = 4
    # Evaluate the finetuned planners your run SAVED: steps 3,6,...,90 (Planner_{round*per_round}.pt).
    # Override via env vars: ODP_EVAL_CKPT (single step) or ODP_EVAL_FROM/ODP_EVAL_TO/ODP_EVAL_BY.
    checkpoint = int(os.environ.get('ODP_EVAL_FROM', os.environ.get('ODP_EVAL_CKPT', 90)))
    _ckpt_to = int(os.environ.get('ODP_EVAL_TO', os.environ.get('ODP_EVAL_CKPT', checkpoint))) + 1
    _ckpt_by = int(os.environ.get('ODP_EVAL_BY', 3))
    _n_seeds = int(os.environ.get('ODP_EVAL_SEEDS', 50))
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
    chunk_size = [31, 25, 20, 19, 18, 13, 12, 11, 10, 15, 7, 6, 8, 5, 16, 4, 9, 14, 17, 21, 22, 23, 24, 26, 27, 28, 29, 30]
    #for seed in [10001, 20002, 30003, 40004, 50005, 60006, 70007, 80008, 90009, 100010, 110011, 120012]:
    set_seed(1)

    # Eval-speed knobs (env vars): ODP_EVAL_EPLEN shortens episodes (default 3000), ODP_EVAL_MAXCHUNKS caps
    # the per-seed chunk_size retries (default = all). A FAILING planner exhausts every chunk_size x a full
    # episode per seed, so these bound the wall-clock for a quick read.
    _eplen = int(os.environ.get('ODP_EVAL_EPLEN', 3000))
    _max_chunks = int(os.environ.get('ODP_EVAL_MAXCHUNKS', len(chunk_size)))
    while(checkpoint < _ckpt_to):
         print(f"Running checkpoing: {checkpoint}")
         total_return = 0.0
         for j in range(1, _n_seeds + 1):
           return_value = 0.0
           chunk_size_index = 0
           while((return_value != 1.0) and (chunk_size_index < min(_max_chunks, len(chunk_size)))):
              return_value, _ = rollout(
                  env_name,
                  specific_train_dataset,
                  horizon,
                  steps_T = 10,
                  num_karras = 1,
                  eta = 0.0,
                  episode_length = _eplen,
                  checkpoint_steps = checkpoint,
                  render = False,
                  base_seed = j,
                  #goal_cell = np.array([6, 1], dtype = int),
                  task_id = task_id,
                  continual_rollout = True,
                  chunk_size = chunk_size[chunk_size_index],
                  #chunk_size = 1,
                  device = device)
              chunk_size_index += 1
           print(f"  seed {j}: {return_value} (tried {chunk_size_index} chunk sizes)")
           total_return += return_value
         print(f"Checkpoint: {checkpoint} Success Rate: {total_return / _n_seeds :.4f}")
         checkpoint += _ckpt_by


    

    """
    horizon = 80
    env_name = 'ogpointmaze'
    specific_train_dataset = 'medium'
    task_id = 1
    checkpoint = 0
    total_reward = 0.0
    device = check_device()
    print(f"Using device {device}, checkpoint: {checkpoint}")
    #for i in range(1, 51):
    
    set_seed(2)
    #total_return = 0
    #for j in range(0, 51):
    return_value, steps = rollout(
               env_name, 
               specific_train_dataset, 
               horizon, 
               steps_T = 10, 
               num_karras = 1, 
               eta = 0.0, 
               episode_length = 3000, 
               checkpoint_steps = checkpoint, 
               render = True,  
               base_seed = 1, 
               #goal_cell = np.array([6, 1], dtype = int), 
               task_id = task_id,
               continual_rollout = True,
               chunk_size = 80,
               device = device)
        #total_return += return_value
    print(return_value)
    print(steps)
    exit()
    
      #print(f"seed {i} finished")
    #print(f"Success Rate: {total_reward / 50 :.4f}")
    #print(get_normalized_score(total_reward/10, min_score, max_score))
    print(total_return/50)
    """


    """
    from Finetuning.utils import rollout_parallel3
    #set_seed(1)
    horizon = 32
    env_name = 'cube'
    specific_train_dataset = 'single-play'
    task_id = 4
    total_success_trajs = []
   
    
    #for i in range(30, 60):
       #checkpoint_step = i*5
       #total_success_rate = 0.0
    for i in range(1, 121):
          set_seed(i)
          trajs, _, success_rate, _ = rollout_parallel3(env_name = env_name, 
                      specific_env = specific_train_dataset, 
                      horizon = horizon,
                      steps_T = 200, 
                      num_karras = 10, 
                      eta = 0.8, 
                      episode_length = 4000, 
                      checkpoint_step = 0,
                      num_envs = 10, 
                      task_id = task_id, 
                      seed_base = 1, 
                      continual_rollout = True, 
                      chunk_size = 31)
          success_trajs = get_success_trajs(trajs)
          total_success_trajs.extend(success_trajs)
          exit()
          #success_rate = len(success_trajs) / len(trajs)
          #total_success_rate += success_rate
       #total_success_rate = total_success_rate / 50
      # print(f"Success Rate for checkpoint {checkpoint_step}: {total_success_rate:.4f}")
     # print(len(total_success_trajs))
    #save_success_trajs_for_reward(total_success_trajs, env_name, specific_train_dataset, task_id = 4)
    """


 
    



