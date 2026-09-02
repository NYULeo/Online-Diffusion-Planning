import chunk
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
from Finetuning.utils import (
    cycle,
    get_planner,
    PlannerDataset,
    get_current_state,
    reward_processor,
    check_device,
    #set_seed,
    #configure_precision,
)
from Pretrain.Dataset import Planner_Processor, get_dataset
from Pretrain.Planners.Backbone.Sampler import sample_reverse_sde, sample_euler_karras, sample_euler_karras2
import pickle
import random
import gymnasium as gym
import gymnasium_robotics
from Pretrain.Dataset import get_dataset
from gymnasium.wrappers import TimeLimit
from typing import Optional, List
from dataclasses import dataclass
from typing import List
from Finetuning.traj_reward5 import TotalReward_Critic, RewardConfig, TotalReward
#from Finetuning.Raw import Selector

class Selector:
    def __init__(
        self,
        env_name,
        specific_env,
        RConfig: RewardConfig,
        reward_checkpoint: int,
        kernel_checkpoint: int,
        critic_checkpoint: Optional[int] = None,
        task_id: Optional[int] = None,
        lam: float = 0.0,
        n_candidates: int = 30,
    ):
        self.env_name = env_name
        self.specific_env = specific_env
        self.RConfig = RConfig
        self.task_id = task_id
        self.lam = lam
        self.n_candidates = n_candidates
        self.device = check_device()

        if critic_checkpoint is not None:
            self.model = TotalReward_Critic(
                self.device,
                RConfig,
                env_name,
                specific_env,
                reward_checkpoint,
                kernel_checkpoint,
                critic_checkpoint,
                task_id,
            )
        else:
            self.model = TotalReward(
                self.device,
                RConfig,
                env_name,
                specific_env,
                reward_checkpoint,
                kernel_checkpoint,
                task_id,
            )
        self.model.eval()

    def select_plan(self, plans: List[np.ndarray]) -> np.ndarray:
        if len(plans) == 0:
            raise ValueError("select_plan received an empty plan list")

        rewards = []
        with torch.no_grad():
            for plan in plans:
                if isinstance(plan, torch.Tensor):
                    plan_tensor = plan.detach().float().to(self.device)
                else:
                    plan_np = np.ascontiguousarray(plan, dtype=np.float32)
                    plan_tensor = torch.from_numpy(plan_np).to(self.device)
                reward = self.model.predict(plan_tensor, self.lam)
                rewards.append(float(reward.detach().cpu()))

        return np.asarray(plans[int(np.argmax(rewards))], dtype=np.float32).copy()



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


def rollout(env_name, 
            specific_env, 
            horizon, 
            num_layers, 
            steps_T, 
            num_karras, 
            eta, 
            episode_length, 
            checkpoint_steps, 
            render = False, 
            goal_cell: Optional[np.ndarray] = None, 
            start_cell: Optional[np.ndarray] = None, 
            task_id: Optional[int] = None, 
            base_seed: int = 0, 
            continual_rollout = False, 
            chunk_size = 5, 
            device = None, 
            selector: Optional[Selector] = None):
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
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= num_layers, timestep_emb_type="fourier").to(device)
     elif (env_name == 'pointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= num_layers, timestep_emb_type="fourier").to(device)
     elif(env_name == 'antmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= num_layers, timestep_emb_type="fourier").to(device)
     elif(env_name == 'cube'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= num_layers, timestep_emb_type="fourier").to(device)
     elif(env_name == 'ogpointmaze'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= num_layers, timestep_emb_type="fourier").to(device)
     elif(env_name == 'scene'):
           model = DiT1d(in_dim = (d_s + d_a), emb_dim = 128, d_model = 256, n_heads = 256//64, depth= num_layers, timestep_emb_type="fourier").to(device)
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
                         """
                         x = selector.sample_selected_plan(
                                current_state_norm, model, d_s, d_a, horizon,
                                steps_T, num_karras, eta, device,
                            )
                         """
                         Plans = [
                               sample_euler_karras(current_state_norm, model, d_s, d_a, horizon, steps_T, num_karras, eta, device)
                               for _ in range(selector.n_candidates)
                            ]
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
                    
                    x = selector.sample_selected_plan(
                                current_state_norm, model, d_s, d_a, horizon,
                                steps_T, num_karras, eta, device,
                            )

                action = x[0, d_s:(d_s+d_a)].copy()
                generated_state = x[1, :d_s].copy()
                action = np.clip(action, -1.0, 1.0)
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
                break
           

           """
           if(terminated):
                break
           """
           
        
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
     #print(traj['rewards'])
     #return traj
     
     #return rewards[-1], len(observations)
     return float(info['success']), len(observations)
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
    checkpoint = 90
    total_reward = 0.0
    device = check_device()
    #configure_precision()
    print(f"Using device {device}")
    total_return = 0.0
    
    RConfig = RewardConfig(
                    beta=1.0,
                    min_log_prob=-110.0,
                    quantile=0.999,
                    critic_gamma=0.99,
                    explore=False,
                    type_kernel='mog',
                    kernel_num_modes=10,
                    kernel_noise_floor=5e-4,
                    num_hidden_layers_kernel=4,
                    hidden_dim_kernel=514,
                    num_hidden_layers_reward=4,
                    hidden_dim_reward=512,
                    num_hidden_layers_critic=4,
                    hidden_dim_critic=512,
            )
    
    """
    set_seed(1)
    selector = Selector(
                env_name,
                specific_train_dataset,
                RConfig,
                reward_checkpoint=0,
                kernel_checkpoint=0,
                critic_checkpoint=checkpoint,   # omit or None to use TotalReward only
                task_id=task_id,
                lam=0.0,
                n_candidates=50,
            )
    
    return_value, length = rollout(
            env_name,
            specific_train_dataset,
            horizon,
            num_layers = 2,
            steps_T = 10,
            num_karras = 1,
            eta=0.0,
            episode_length=5000,
            checkpoint_steps=checkpoint,
            render=True,
            base_seed=1,
            task_id=task_id,
            continual_rollout=True,
            chunk_size = 15,
            device=device,
            selector=selector,
          )
   # print(length)
    exit()
    """
    #set_seed(1)
    selector = Selector(
                env_name,
                specific_train_dataset,
                RConfig,
                reward_checkpoint=0,
                kernel_checkpoint=0,
                critic_checkpoint=checkpoint,   # omit or None to use TotalReward only
                task_id=task_id,
                lam=0.0,
                n_candidates=50,
            )
    total = 0.0
    set_seed(1)
    for i in range(1, 101):
         #set_seed(i)
         return_value, length = rollout(
            env_name,
            specific_train_dataset,
            horizon,
            num_layers=2,
            steps_T=10,
            num_karras=1,
            eta=0.0,
            episode_length=5000,
            checkpoint_steps=checkpoint,
            render=False,
            base_seed = i,
            task_id=task_id,
            continual_rollout=True,
            chunk_size=15,
            device=device,
            selector=selector,
          )
         print(return_value)
         total += return_value
         #print()
    print(f"Success Rate: {total / 100 :.4f}")
    exit()

    
    



