'''Online finetuning orchestrator (JAX/Flax port).

Single-device orchestration of the adjoint-matching planner finetuning loop: build the buffer datasets,
construct the reward/critic models and the adjoint-matching finetuner, then alternate finetuning rounds
with rollouts and reward/kernel/critic retraining. The heavy lifting is delegated to the converted
sibling modules (``Acc_AdjointMatchingFineTuner``, ``rollout_parallel2``, ``train_*``, ``sample_euler_karras``).

The original torch version used HuggingFace ``accelerate`` for multi-device data parallelism. This JAX
port is single-device (JAX places arrays automatically); the multi-process collectives are replaced by a
small in-file single-device shim ``_SingleDeviceAccelerator`` so every ``self.accelerator.*`` call site
stays faithful (main process is always true, gather is identity, split yields the whole input).
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from dataclasses import dataclass
from gymnasium.vector import AsyncVectorEnv
from Finetuning.utils import Lambda, RewardDataset, PlannerDataset, KernelDataset, cycle, EMA, RewardTracker, get_trajs, get_success_trajs, check_Critic, get_kernel, get_new_critic_stats, load_success_trajs
#from Finetuning.traj_reward import RewardConfig, TotalReward, TotalReward_Critic
from Finetuning.traj_reward import RewardConfig, TotalReward, TotalReward_Critic
from Finetuning.adjoint_matching import AdjointMatchingFineTuner, AdjointMatchingConfig
from Finetuning.acc_adjoint_matching import Acc_AdjointMatchingConfig, Acc_AdjointMatchingFineTuner
#from AM import Acc_AdjointMatchingConfig, Acc_AdjointMatchingFineTuner
from Finetuning.Rollout import rollout
from Pretrain.Planners.Backbone.Dit import DiT1d
from Pretrain.Dataset import get_PlannerName, get_dataset, Planner_Processor
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras
from typing import List
from Finetuning.utils import TrajectoryDict, rollout_parallel, get_planner, rollout_parallel2, save_planner, train_reward, train_kernel, train_kernel_mog, train_critic, save_trajs, AlphaSchedulerConfig, checktrajs, rollout_parallel3, train_reward_ensemble
from Pretrain.Dataset import get_env

import jax
import jax.numpy as jnp
import numpy as np
import copy
import os
import math
from typing import Optional, Dict
import json
from dataclasses import asdict
from random import random
import random


# ----------------------------------------------------------------------------------------------------------------------
# Single-device replacement for HuggingFace `accelerate.Accelerator`.
#
# The torch orchestrator used `accelerate` purely for multi-process data parallelism (gather/broadcast/
# split-between-processes/wait-for-everyone) plus device/mixed-precision bookkeeping. JAX is single-device
# here, so all collectives collapse to identities and `is_main_process` is always True. This keeps every
# `self.accelerator.*` call site byte-for-byte identical to the torch version.
# ----------------------------------------------------------------------------------------------------------------------
class _GatheredArray(np.ndarray):
    '''numpy-array view that also answers torch-style `.float()`/`.int()` so single-device gather output
    can be consumed by the original orchestration arithmetic (`.float()[:n]`, `.int().sum().item()`).'''

    def float(self):
        return self.astype(np.float32)

    def int(self):
        return self.astype(np.int64)


class _SingleDeviceAccelerator:
    '''Minimal single-device stand-in for `accelerate.Accelerator` (no torch, no multi-process).'''

    def __init__(self, mixed_precision=None, gradient_accumulation_steps=1):
        # API-CHANGE: accelerate.Accelerator replaced by a single-device shim (CONVERSION_GUIDE §6 — drop
        # accelerate for single-device). Args retained for call-site faithfulness; ignored here.
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = None  # JAX places arrays automatically (CONVERSION_GUIDE §9).
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0
        self.sync_gradients = True  # single device always "syncs" (no grad accumulation across procs)

    def wait_for_everyone(self):
        pass

    def gather_for_metrics(self, data, use_gather_object=False):
        if use_gather_object:
            # Object path: torch returned the gathered list; single-device => the input list unchanged.
            return data
        # Tensor path: torch gathered scalar tensors into one tensor. Single-device => wrap as array view.
        return np.asarray(data, dtype=np.float32).view(_GatheredArray)

    def split_between_processes(self, inputs):
        # Single process owns the whole input. Mirror accelerate's context-manager contract.
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield inputs

        return _cm()

    def prepare(self, *args):
        # accelerate.prepare wraps models/optimizers/dataloaders for distributed training; on a single
        # device it's a passthrough. Returns a tuple matching the inputs (or the single object).
        return args[0] if len(args) == 1 else args

    def unwrap_model(self, model):
        # No DDP/wrapping on a single device.
        return model

    def reduce(self, tensor, reduction='mean'):
        # Nothing to reduce across a single process.
        return tensor

    def autocast(self):
        # No mixed-precision autocast (JAX/optax handles dtypes); no-op context manager.
        from contextlib import nullcontext
        return nullcontext()

    def accumulate(self, *models):
        # Gradient accumulation is handled in the optax chain, not here; no-op context manager.
        from contextlib import nullcontext
        return nullcontext()


@dataclass
class Train_Reward_Config:
    hidden_layers: int = 1
    hidden_dim: int = 128
    ensemble_size: Optional[int] = None
    batch_size: int = 32
    num_steps: int = 1000
    lr: float = 2e-4
    min_lr: float = 1e-5
    sigma: float = 7.0
    target_reward: Optional[float] = None
    train_goal: Optional[np.ndarray] = None
    task_id: Optional[int] = None
    rollout_goal: Optional[np.ndarray] = None
    rollout_start_cells: Optional[np.ndarray] = None

@dataclass
class Train_Kernel_Config:
    batch_size: int = 256
    num_steps: int = 1000
    lr: float = 1e-3
    ensemble_size: int = 10
    num_hidden_layers: int = 2
    hidden_dim: int = 256
    type_kernel: str = 'robust' or 'mog'
    kernel_num_modes: Optional[int] = 8
    kernel_noise_floor: Optional[float] = 1e-4
    λ_reg: float = 1e-3

@dataclass
class Train_Critic_Config:
    hidden_layers: int = 2
    hidden_dim: int = 128
    batch_size: int = 256
    num_steps: int = 3000
    lr: float = 5e-05
    min_lr: float = 1e-05
    tau: float = 0.005
    gamma: float = 1.0
    lam: float = 0.95
    horizon: int = 200
    data_conservation: bool = False
    momentum: float = 0.005

@dataclass
class FinetuningConfig():
    AMConfig: AdjointMatchingConfig | Acc_AdjointMatchingConfig
    RewardConfig: RewardConfig
    AlphaConfig: AlphaSchedulerConfig
    dataset_name: str
    specific_dataset: str
    planner_checkpoint: int
    reward_model_checkpoint: int
    kernel_model_checkpoint: int
    critic_model_checkpoint: int
    train_reward_config: Train_Reward_Config
    train_kernel_config: Train_Kernel_Config
    train_critic_config: Train_Critic_Config
    offline: bool = False
    critic: bool = False
    update_critic: bool = True
    kernel: bool = True
    update_kernel: bool = True
    buffer_size: int = 100000
    finetune_buffer_cutoff_length: Optional[int] = None
    train_buffer_cutoff_length: Optional[int] = None
    finetune_steps: int = 1000000
    finetune_rounds: int = 10
    diffusion_steps: int = 30
    karras_percent: float = 0.05
    Loss_Clip_percent: float = 0.75
    finetune_batch_size: int = 12
    finetune_batch_per_sample: int = 3
    finetune_lr: float = 1e-6
    initial_lam: float = 0.01
    eta_lam: float = 0.001
    gradient_accumulate_every: int = 1
    update_lambda_every: int = 5
    reward_scaling_factor: float = 100000
    MaxEnt: bool = False
    Entropy_Scaling_Factor: float = 0.5
    rollout_length: int = 1000
    rollout_num_envs: int = 1
    num_rollout_processes: Optional[int] = None
    continual_rollout: bool = False
    chunk_size: int = 10

def save_hyperparameters(config: FinetuningConfig, filepath: Optional[str] = None):
    if filepath is None:
        os.makedirs(f"./Finetuning/args/{config.dataset_name}/{config.specific_dataset}/", exist_ok=True)
        filepath = f"./Finetuning/args/{config.dataset_name}/{config.specific_dataset}/hyperparameters.json"

    def convert_to_json_serializable(obj):
        """Recursively convert objects to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        # API-CHANGE: torch.device branch dropped (torch removed). jax devices stringify via the generic
        # __dict__/str fallback below; numeric/None paths are unchanged.
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
            # Handle other custom objects by converting to string
            return str(obj)
        return obj

    # Convert all config dataclasses to dictionaries
    hyperparams = {
        'env_details': {
            'dataset_name': config.dataset_name,
            'specific_dataset': config.specific_dataset,
        },
        'pretrained_models': {
            'planner_checkpoint': config.planner_checkpoint,
            'reward_model_checkpoint': config.reward_model_checkpoint,
            'kernel_model_checkpoint': config.kernel_model_checkpoint,
            'critic_model_checkpoint': getattr(config, 'critic_model_checkpoint', None),
        },
        'adjoint_matching_config': convert_to_json_serializable(asdict(config.AMConfig)),
        'reward_config': convert_to_json_serializable(asdict(config.RewardConfig)),
        'alpha_config': convert_to_json_serializable(asdict(config.AlphaConfig)),
        'finetuning_hyperparameters': {
            'finetune_batch_size': config.finetune_batch_size,
            'finetune_batch_per_sample': config.finetune_batch_per_sample,
            'buffer_size': config.buffer_size,
            'finetune_buffer_cutoff_length': config.finetune_buffer_cutoff_length,
            'train_buffer_cutoff_length': config.train_buffer_cutoff_length,
            'finetune_lr': config.finetune_lr,
            'reward_scaling_factor': config.reward_scaling_factor,
            'finetune_total_steps': config.finetune_steps,
            'finetune_rounds': config.finetune_rounds,
            'diffusion_steps': config.diffusion_steps,
            'karras_percent': config.karras_percent,
            'Loss_Clip_percent': config.Loss_Clip_percent,
            'gradient_accumulate_every': config.gradient_accumulate_every,
            'initial_lam': config.initial_lam,
            'eta_lam': config.eta_lam,
            'update_lambda_every': config.update_lambda_every,
            'MaxEnt': config.MaxEnt,
            'Entropy_Scaling_Factor': config.Entropy_Scaling_Factor
        },

        'exploration_hyperparameters': {
            'rollout_length': config.rollout_length,
            'rollout_num_envs': config.rollout_num_envs,
            'num_rollout_processes': config.num_rollout_processes,
            'continual_rollout': config.continual_rollout,
            'rollout_goal': config.train_reward_config.rollout_goal.tolist() if hasattr(config.train_reward_config.rollout_goal, 'tolist') else config.train_reward_config.rollout_goal,
        },

        'model_updates': {
             'critic': config.critic,
             'kernel': config.kernel,
        },

        'reward_training': convert_to_json_serializable(asdict(config.train_reward_config)),
        'kernel_training': convert_to_json_serializable(asdict(config.train_kernel_config)),
        'critic_training': convert_to_json_serializable(asdict(config.train_critic_config)) if config.critic else None,
        'device_info': {
            'device': str(config.AMConfig.device),
            # API-CHANGE: torch.cuda.* device queries replaced by jax.devices() (torch removed).
            'num_gpus': len(jax.devices()),
            'gpu_name': str(jax.devices()[0]) if len(jax.devices()) > 0 else None,
        }
    }

    # Handle numpy arrays, jax devices, and other non-JSON-serializable types
    hyperparams = convert_to_json_serializable(hyperparams)

    # Save with pretty printing (indent=4 makes it human-readable)
    with open(filepath, 'w') as f:
        json.dump(hyperparams, f, indent=4, sort_keys=False)

    print(f"Hyperparameters saved to {filepath}")

class OnlineFinetuner():
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.config.RewardConfig.num_hidden_layers_kernel = self.config.train_kernel_config.num_hidden_layers
        self.config.RewardConfig.hidden_dim_kernel = self.config.train_kernel_config.hidden_dim
        self.config.RewardConfig.num_hidden_layers_reward = self.config.train_reward_config.hidden_layers
        self.config.RewardConfig.hidden_dim_reward = self.config.train_reward_config.hidden_dim
        self.config.RewardConfig.num_hidden_layers_critic = self.config.train_critic_config.hidden_layers
        self.config.RewardConfig.hidden_dim_critic = self.config.train_critic_config.hidden_dim
        self.config.RewardConfig.type_kernel = self.config.train_kernel_config.type_kernel
        self.config.RewardConfig.kernel_num_modes = self.config.train_kernel_config.kernel_num_modes
        self.config.RewardConfig.kernel_noise_floor = self.config.train_kernel_config.kernel_noise_floor
        self.config.AMConfig.task_id = self.config.train_reward_config.task_id
        self.config.AMConfig.alpha_scheduler_config = self.config.AlphaConfig
        self.config.AMConfig.finetune_total_steps = self.config.finetune_steps
        self.config.AMConfig.batch_per_sample = self.config.finetune_batch_per_sample
        self.config.AMConfig.per_round_steps = self.config.finetune_steps // self.config.finetune_rounds
        self.config.AMConfig.diffusion_steps = self.config.diffusion_steps
        self.config.AMConfig.num_karras = math.ceil(self.config.diffusion_steps * self.config.karras_percent)
        self.config.AMConfig.num_Loss_Clip_steps = math.ceil(self.config.diffusion_steps * self.config.Loss_Clip_percent)
        self.config.AMConfig.dataset_name = self.config.dataset_name
        self.config.AMConfig.specific_dataset = self.config.specific_dataset
        self.config.AMConfig.finetune_lr = self.config.finetune_lr
        self.env, d_s, d_a = get_env(self.config.dataset_name, self.config.specific_dataset)
        self.config.AMConfig.d_s = d_s
        self.config.AMConfig.d_a = d_a
        self.config.AMConfig.update_kernel = self.config.kernel
        self.config.AMConfig.eta_lam = self.config.eta_lam
        if self.config.AMConfig.update_kernel:
             self.config.AMConfig.lam = self.config.initial_lam
        else:
             self.config.AMConfig.lam = 0.0
        #self.config.AMConfig.update_ema_every = self.config.update_lambda_every
        self.config.AMConfig.update_ema_every = self.config.gradient_accumulate_every
        self.config.AMConfig.reward_scaling_factor = self.config.reward_scaling_factor
        self.config.AMConfig.update_lambda_every = self.config.update_lambda_every
        self.config.AMConfig.MaxEnt = self.config.MaxEnt
        self.config.AMConfig.Entropy_Scaling_Factor = self.config.Entropy_Scaling_Factor

        #self.accelerator = Accelerator(mixed_precision = 'bf16')
        self.accelerator = _SingleDeviceAccelerator(
               mixed_precision='bf16',
               gradient_accumulation_steps = self.config.gradient_accumulate_every,
        )
        self.device = self.accelerator.device

        self.Initialize_BufferDataset()
        self.set_reward_model(self.device)
        self.AMFineTuner = Acc_AdjointMatchingFineTuner(
                   self.accelerator,
                   self.config.planner_checkpoint,
                   self.config.AMConfig)

    def Initialize_BufferDataset(self):
        self.Finetune_Buffer = []
        self.Train_Buffer = []
        self.Train_Kernel_Buffer = []
        if(self.config.critic and self.config.update_critic):
             self.Base_Critic_Buffer = []
        else:
             self.Base_Critic_Buffer = None

        if(self.config.train_reward_config.task_id is not None):
            dataset_reward = get_dataset(self.config.dataset_name, self.config.specific_dataset, task_id = self.config.train_reward_config.task_id, traj_length = self.config.train_buffer_cutoff_length)
            trajs_reward = dataset_reward.get_trajectories()
            dataset_kernel = get_dataset(self.config.dataset_name, self.config.specific_dataset, task_id = self.config.train_reward_config.task_id)
            trajs_kernel = dataset_kernel.get_trajectories()
            self.Finetune_Buffer.extend(trajs_reward)
            self.Train_Buffer.extend(trajs_reward)
            self.Train_Kernel_Buffer.extend(trajs_kernel)
            if(self.Base_Critic_Buffer is not None):
                #success_trajs = load_success_trajs(self.config.dataset_name, self.config.specific_dataset, self.config.train_reward_config.task_id, step = 0)
                #self.Base_Critic_Buffer.extend(success_trajs)
                self.Base_Critic_Buffer.extend(trajs_reward)

        elif(self.config.train_reward_config.train_goal is not None):
            dataset_reward = get_dataset(self.config.dataset_name, self.config.specific_dataset, goal = self.config.train_reward_config.train_goal, mode = 'reward')
            trajs_reward = dataset_reward.get_trajectories()
            dataset_kernel = get_dataset(self.config.dataset_name, self.config.specific_dataset)
            trajs_kernel = dataset_kernel.get_trajectories()
            dataset_critic = get_dataset(self.config.dataset_name, self.config.specific_dataset, goal = self.config.train_reward_config.train_goal, mode = 'critic')
            trajs_critic = dataset_critic.get_trajectories()
            self.Finetune_Buffer.extend(trajs_reward)
            self.Train_Buffer.extend(trajs_reward)
            self.Train_Kernel_Buffer.extend(trajs_kernel)
            if(self.Base_Critic_Buffer is not None):
                self.Base_Critic_Buffer.extend(trajs_critic)
                #self.Base_Critic_Buffer.extend(trajs_reward)

        else:
            dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset)
            trajs = dataset.get_trajectories()
            self.Finetune_Buffer.extend(trajs)
            self.Train_Buffer.extend(trajs)
            self.Train_Kernel_Buffer.extend(trajs)
            if(self.Base_Critic_Buffer is not None):
                self.Base_Critic_Buffer.extend(trajs)

        self.PlannerDataset = PlannerDataset(
                   self.Finetune_Buffer,
                   self.config.AMConfig.horizon,
                   self.config.dataset_name,
                   self.config.specific_dataset,
                   self.config.train_reward_config.task_id,
                   self.config.finetune_buffer_cutoff_length)

    def set_reward_model(self, device):
        if (not self.config.critic) :
            self.reward_model = TotalReward(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint, self.config.train_reward_config.task_id)
        else:
            self.reward_model = TotalReward_Critic(device, self.config.RewardConfig, self.config.dataset_name, self.config.specific_dataset, self.config.reward_model_checkpoint, self.config.kernel_model_checkpoint, self.config.critic_model_checkpoint, self.config.train_reward_config.task_id)

    def gather_and_sync_trajs_and_buffer(self, local_trajs):
        # Gather local trajectories from all processes
        gathered_trajs_list = self.accelerator.gather_for_metrics([local_trajs if local_trajs else []], use_gather_object=True)
        self.accelerator.wait_for_everyone()

        update_reward = False
        if self.accelerator.is_main_process:
            # Flatten and extend buffer only on main
            collected_trajs = []
            for process_trajs in gathered_trajs_list:
               if process_trajs:
                  collected_trajs.extend(process_trajs)

            num_rollout = (self.config.num_rollout_processes
               if self.config.num_rollout_processes is not None
               else self.accelerator.num_processes)
            print(f"Rollout Completed: Collected {len(collected_trajs)} trajectories across {num_rollout} rollout processes")
            #print(f"Rollout Completed: Collected {len(collected_trajs)} trajectories across {self.accelerator.num_processes} processes")

            self.Train_Kernel_Buffer.extend(collected_trajs)
            """
            success_trajs = get_success_trajs(collected_trajs)
            if(len(success_trajs) > 0):
                 self.Train_Buffer.extend(success_trajs)
                 self.Finetune_Buffer.extend(success_trajs)
                 update_reward = True
            """
            self.Train_Buffer.extend(collected_trajs)
            self.Finetune_Buffer.extend(collected_trajs)
            update_reward = True
            if len(self.Finetune_Buffer) > self.config.buffer_size:
                 num_to_remove = len(self.Finetune_Buffer) - self.config.buffer_size
                 self.Finetune_Buffer = self.Finetune_Buffer[num_to_remove:]
                 #print(f"Buffer size limited to {self.config.buffer_size}, removed {num_to_remove} oldest trajectories")

            # Prepare updated buffer for sync
            buffer_for_sync = [self.Finetune_Buffer]
        else:
            buffer_for_sync = [None]

        # Broadcast full updated buffer to all processes
        synced_buffer = self.accelerator.gather_for_metrics(buffer_for_sync, use_gather_object=True)
        if synced_buffer[0] is not None:
             self.Finetune_Buffer = synced_buffer[0]
        # Single-device: broadcast of the update flag from the main process is the identity.
        flag = np.asarray([1 if update_reward else 0], dtype=np.int64)
        update_reward = bool(flag.item())

        self.PlannerDataset = PlannerDataset(
                 self.Finetune_Buffer,
                 self.config.AMConfig.horizon,
                 self.config.dataset_name,
                 self.config.specific_dataset,
                 self.config.train_reward_config.task_id,
                 self.config.finetune_buffer_cutoff_length
         )
        return update_reward

    def collect_critic_buffer(self, local_trajs):
          # ALL processes must participate in gather_for_metrics (collective operation)
          gathered_trajs_list = self.accelerator.gather_for_metrics([local_trajs if local_trajs else []], use_gather_object=True)
          self.accelerator.wait_for_everyone()
          update_critic = False
          # Only main process needs to process the gathered data
          if self.accelerator.is_main_process:
              total_trajs = []
              for process_trajs in gathered_trajs_list:
                  if process_trajs:
                      total_trajs.extend(process_trajs)
              """
              success_trajs = get_success_trajs(total_trajs)
              if(len(success_trajs) > 1):
                  update_critic = True
              if self.config.train_critic_config.data_conservation:
                  critic_buffer = self.data_conservation_update(success_trajs)
              else:
                  critic_buffer = success_trajs
              self.Base_Critic_Buffer.extend(success_trajs.copy())
              """
              update_critic = True
              success_trajs = get_success_trajs(total_trajs)
              if self.config.train_critic_config.data_conservation:
                  critic_buffer = self.data_conservation_update(total_trajs)
              else:
                  critic_buffer = total_trajs
              self.Base_Critic_Buffer.extend(total_trajs.copy())
              critic_buffer.extend(success_trajs)
          else:
              critic_buffer = None

          return critic_buffer, update_critic

    def data_conservation_update(self, critic_buffer):
        """
        if(self.config.train_reward_config.task_id is not None):
            dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset, task_id = self.config.train_reward_config.task_id, traj_length = self.config.train_buffer_cutoff_length)
            trajs = dataset.get_trajectories()
        elif(self.config.train_reward_config.train_goal is not None):
            dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset, goal = self.config.train_reward_config.train_goal, mode = 'critic')
            trajs = dataset.get_trajectories()
        else:
            dataset = get_dataset(self.config.dataset_name, self.config.specific_dataset)
            trajs = dataset.get_trajectories()
        """
        trajs = self.Base_Critic_Buffer.copy()

        if(len(critic_buffer) < 2):
             critic_buffer.extend(trajs)
        else:
             half_size_1 = len(trajs) // 2
             half_pretrained_trajs = random.sample(trajs, half_size_1)
             half_size_2 = len(critic_buffer) // 2
             half_buffer_trajs = random.sample(critic_buffer, half_size_2)
             critic_buffer = half_pretrained_trajs + half_buffer_trajs
        return critic_buffer

    """
    def get_generated_plans(self, number_of_generated_plans: int):
        dataloader = cycle(DataLoader(self.PlannerDataset, batch_size = 12, shuffle = False))
        generated_plans = []
        for i in range(number_of_generated_plans):
            s0 = next(dataloader)
            s0 = s0.squeeze(0).cpu().numpy()
            x = sample_euler_karras(s0,
                               self.AMFineTuner.new_score_net,
                               self.config.AMConfig.d_s,
                               self.config.AMConfig.d_a,
                               self.config.AMConfig.horizon,
                               self.config.AMConfig.diffusion_steps,
                               self.config.AMConfig.num_karras,
                               self.config.AMConfig.eta,
                               self.device)

            generated_plans.append(x)
        return generated_plans
    """

    def get_generated_plans(self, number_of_generated_plans: int, *, rng=None):
         # Build a deterministic (unshuffled) s0 batch from the planner dataset. Datasets are numpy-backed
         # (CONVERSION_GUIDE §13), so iterate directly instead of via a torch DataLoader.
         n = min(number_of_generated_plans, len(self.PlannerDataset))
         s0_batch = [np.asarray(self.PlannerDataset[i]) for i in range(n)]
          # Optional safety: trim/pad logic if dataset is smaller than N
         s0_batch = s0_batch[:number_of_generated_plans]

          #    Split s0 list across processes (single-device => whole batch)
         with self.accelerator.split_between_processes(s0_batch) as local_s0_batch:
            local_generated = []
            for s0 in local_s0_batch:
                 s0_np = np.asarray(s0)
                 if rng is not None:
                     rng, sample_key = jax.random.split(rng)
                 else:
                     sample_key = None
                 x = sample_euler_karras(
                    s0_np,
                    self.AMFineTuner.new_score_net,
                    self.config.AMConfig.d_s,
                    self.config.AMConfig.d_a,
                    self.config.AMConfig.horizon,
                    self.config.AMConfig.diffusion_steps,
                    self.config.AMConfig.num_karras,
                    self.config.AMConfig.eta,
                    rng=sample_key,
                  )
                 local_generated.append(x)  # each x: np.ndarray (H, d_s+d_a)

         self.accelerator.wait_for_everyone()

         # Gather python lists from all ranks (single-device => identity)
         gathered = self.accelerator.gather_for_metrics(
                [local_generated], use_gather_object=True
         )

         generated_plans = []
         for per_rank in gathered:
                generated_plans.extend(per_rank)

         return generated_plans[:number_of_generated_plans]

    def _sample_planner_batch(self, batch_size, *, rng, shuffle=True, drop_last=True):
         # numpy-backed mini-batch generator over the planner conditions, replacing the torch DataLoader +
         # DistributedSampler. INFINITE: the AM finetuner consumes `per_round_steps` batches per round via
         # next(dataloader), which exceeds one epoch — so loop forever, reshuffling each epoch (a one-shot
         # generator wrapped in cycle() would hang once exhausted).
         num = len(self.PlannerDataset)
         conditions = jnp.stack([jnp.asarray(self.PlannerDataset[i]) for i in range(num)], axis=0)
         num_batches = num // batch_size if drop_last else math.ceil(num / batch_size)
         num_batches = max(num_batches, 1)   # never 0 (would make the loop spin without yielding -> hang)
         while True:
             rng, perm_key = jax.random.split(rng)
             order = jax.random.permutation(perm_key, num) if shuffle else jnp.arange(num)
             for b in range(num_batches):
                 idx = order[b * batch_size:(b + 1) * batch_size]
                 yield conditions[idx]

    def finetune_planner(self, *, seed=None):
        rng = jax.random.PRNGKey(0) if seed is None else jax.random.PRNGKey(seed)
        if self.accelerator.is_main_process:
            print("Env Details: ------------------------------------------------------------------------------")
            print(f"env_name: {self.config.dataset_name}")
            print(f"specific_env: {self.config.specific_dataset}")
            print('Pretrained Model Details: -----------------------------------------------------------------')
            print(f"planner_checkpoint: {self.config.planner_checkpoint}")
            print(f"reward_model_checkpoint: {self.config.reward_model_checkpoint}")
            print(f"kernel_model_checkpoint: {self.config.kernel_model_checkpoint}")
            print('Finetuning Hyperparameters: ---------------------------------------------------------------')
            print(f"Offline: {self.config.offline}")
            print(f"finetune_batch_size: {self.config.finetune_batch_size}")
            print(f"finetune_batch_per_sample: {self.config.finetune_batch_per_sample}")
            print(f"Finetuning Buffer Size: {self.config.buffer_size}")
            print(f"finetune_lr: {self.config.AMConfig.finetune_lr}")
            print(f"reward_scaling_factor: {self.config.AMConfig.reward_scaling_factor}")
            print(f"finetune_total_steps: {self.config.AMConfig.finetune_total_steps}")
            print(f"finetuning rounds: {self.config.finetune_rounds}")
            print(f"diffusion_steps: { self.config.AMConfig.diffusion_steps}")
            print(f"karras steps: {self.config.AMConfig.num_karras}")
            print(f"Loss clip steps: {self.config.AMConfig.num_Loss_Clip_steps}")
            print(f"gradient accumulate every: {self.config.gradient_accumulate_every}")
            print(f"Initial lambda: {self.config.AMConfig.lam}")
            print(f"eta_lam: {self.config.AMConfig.eta_lam}")
            print(f"update_lambda_every: {self.config.AMConfig.update_lambda_every}")
            print(f"Differential Entropy Maximization:: {self.config.AMConfig.MaxEnt}")
            print(f"Entropy Scaling Factor: {self.config.AMConfig.Entropy_Scaling_Factor}")
            print(f"Exploration Hyperarameters: --------------------------------------------------------------")
            print(f"Rollout Episode Length: {self.config.rollout_length}")
            print(f"Rollout Number of Environments: {self.config.rollout_num_envs}")
            print(f"Rollout Goal Cell: {self.config.train_reward_config.rollout_goal}")
            print(f"Rollout Start Cells: {self.config.train_reward_config.rollout_start_cells}")
            print(f"Critic: {self.config.critic}")
            print(f"Rollout Episode Length: {self.config.rollout_length}")
            print(f"Continual Rollout: {self.config.continual_rollout}")
            print(f"Number of Rollout Processes: {self.config.num_rollout_processes}")
            print('Reward Training Hyperparameters: ----------------------------------------------------------')
            print(f"Reward Hidden Layers: {self.config.train_reward_config.hidden_layers}")
            print(f'Reward Hidden Dim: {self.config.train_reward_config.hidden_dim}')
            print(f"Reward Batch Size: {self.config.train_reward_config.batch_size}")
            print(f"Reward Number of steps: {self.config.train_reward_config.num_steps}")
            print(f"Reward Learning Rate: {self.config.train_reward_config.lr}")
            print(f"Reward Sigma: {self.config.train_reward_config.sigma}")
            print(f"Reward Target Reward: {self.config.train_reward_config.target_reward}")
            print('Kernel Training Hyperparameters: ----------------------------------------------------------')
            print(f'Kernel Hidden Layers: {self.config.train_kernel_config.num_hidden_layers}')
            print(f'Kernel Hidden Dim: {self.config.train_kernel_config.hidden_dim}')
            print(f'Kernel Ensemble Size: {self.config.train_kernel_config.ensemble_size}')
            print(f'Kernel Batch Size: {self.config.train_kernel_config.batch_size}')
            print(f'Kernel Learning Rate: {self.config.train_kernel_config.lr}')
            print(f'Kernel Number of Steps: {self.config.train_kernel_config.num_steps}')
            print(f'Kernel Lambda Regularization: {self.config.train_kernel_config.λ_reg}')
            if(self.config.critic):
                print(f"Critic Training Hyperarameters: ----------------------------------------------------------")
                print(f"Critic Hidden Layers: {self.config.train_critic_config.hidden_layers}")
                print(f"Critic Hidden Dim: {self.config.train_critic_config.hidden_dim}")
                print(f"Critic Model Checkpoint: {self.config.critic_model_checkpoint}")
                print(f"Critic Batch Size: {self.config.train_critic_config.batch_size}")
                print(f"Critic Number of Steps: {self.config.train_critic_config.num_steps}")
                print(f"Critic Learning Rate: {self.config.train_critic_config.lr}")
                print(f"Critic Tau: {self.config.train_critic_config.tau}")
                print(f"Critic Gamma: {self.config.train_critic_config.gamma}")
                print(f"Data Conservation: {self.config.train_critic_config.data_conservation}")
            print('Device Details: ---------------------------------------------------------------------------')
            print(f"The device is: {self.device}")
            # API-CHANGE: torch.cuda.* device queries replaced by jax.devices() (torch removed).
            print(f"The number of GPUs is: {len(jax.devices())}")
            print(f"The GPU name is: {jax.devices()[0] if len(jax.devices()) > 0 else None}")
            print('-------------------------------------------------------------------------------------------')

        if self.accelerator.is_main_process:
             save_hyperparameters(self.config)

        self.accelerator.wait_for_everyone()

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        num_envs_per_process = self.config.rollout_num_envs  # Total envs = base * world_size
        last_critic_update_step = 0
        last_reward_update_step = 0
        for step in range(self.config.finetune_rounds):
            rng, data_key, plan_key = jax.random.split(rng, 3)
            # Single-device: no DistributedSampler; one numpy-backed shuffled pass over the planner dataset.
            dataloader = self._sample_planner_batch(
                self.config.finetune_batch_size,
                rng=data_key,
                shuffle=True,
                drop_last=True)

            if self.accelerator.is_main_process:
                 print(f"Finetuning round {step+1} started")


            #self.AMFineTuner.finetune_planner(dataloader, self.reward_model, step+1)
            self.AMFineTuner.finetune_planner(dataloader, self.reward_model, step+1)
            self.accelerator.wait_for_everyone()


            # (JAX dispatches asynchronously; no explicit device synchronize needed here.)
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                  print(f"Starting Rollout")


            num_rollout_procs = self.config.num_rollout_processes
            do_rollout = (num_rollout_procs is None) or (rank < num_rollout_procs)
            if do_rollout:
                seed_base = rank * num_envs_per_process

                trajs, score, success_rate, total_steps = rollout_parallel2(self.config.dataset_name,
                                             self.config.specific_dataset,
                                             horizon = self.config.AMConfig.horizon,
                                             steps_T = self.config.diffusion_steps,
                                             num_karras = self.config.AMConfig.num_karras,
                                             eta = self.config.AMConfig.eta,
                                             episode_length = self.config.rollout_length,
                                             checkpoint_step = ((step+1) * self.config.AMConfig.per_round_steps),
                                             num_envs = self.config.rollout_num_envs,
                                             goal_cell = self.config.train_reward_config.rollout_goal,
                                             device = self.device,
                                             start_cells = self.config.train_reward_config.rollout_start_cells,
                                             task_id = self.config.train_reward_config.task_id,
                                             seed_base = seed_base,
                                             continual_rollout = self.config.continual_rollout,
                                             chunk_size = self.config.chunk_size)

            else:
                trajs, score, success_rate, total_steps = [], 0.0, 0.0, 0

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                  print(f"Rollout Completed")

            update_reward = self.gather_and_sync_trajs_and_buffer(trajs)
            self.accelerator.wait_for_everyone()

            if self.config.critic:
                 critic_buffer, update_critic = self.collect_critic_buffer(trajs)
                 if self.accelerator.is_main_process:
                     print(f"Number of trajectories for critic training: {len(critic_buffer)}")
                     if(update_critic):
                         print("Training Critic")
                     else:
                         print("Do not Train Critic")
                 self.accelerator.wait_for_everyone()


            #collect the score and number of env stepsacross all processes
            gathered_scores = self.accelerator.gather_for_metrics(np.asarray([score], dtype=np.float32),  use_gather_object=False)
            gathered_success_rates = self.accelerator.gather_for_metrics(np.asarray([success_rate], dtype=np.float32), use_gather_object=False)
            gathered_steps = self.accelerator.gather_for_metrics(np.asarray([total_steps], dtype=np.int64),  use_gather_object=False)
            if self.accelerator.is_main_process:
                 total_steps = gathered_steps.int().sum().item()
                 num_rollout = (num_rollout_procs if num_rollout_procs is not None
                               else self.accelerator.num_processes)
                 rollout_scores = gathered_scores.float()[:num_rollout]
                 rollout_success_rates = gathered_success_rates.float()[:num_rollout]
                 avg_score = rollout_scores.astype(np.float32).mean().item()
                 avg_success_rate = rollout_success_rates.astype(np.float32).mean().item()
                 print(f"Total Number of Environment Steps: {total_steps}")
                 print(f"Average Success Rate: {avg_success_rate:.2f}")
                 print(f"Average Normalized Score: {avg_score:.2f}")
                 # DIAGNOSTIC (read-only, never affects training): the in-loop success metric is
                 # rewards[-1]==1.0, but cube/OGBench episodes don't terminate at the goal, so a planner
                 # that DID reach the goal mid-episode still reads 0. Also report (a) "reached goal at any
                 # step" (max shifted reward hit 1.0 == raw reward hit 0) and (b) torch's distance-to-goal
                 # (final cube pos obs[-1][19:22] vs the task goal). These reveal real task success.
                 try:
                     if self.config.dataset_name == 'cube' and 'single' in self.config.specific_dataset:
                         _goals = {1: np.array([0.0,-1.0,0.199599]), 2: np.array([0.75,0.0,0.199599]),
                                   3: np.array([-0.75,0.0,0.199599]), 4: np.array([0.75,2.0,0.199599]),
                                   5: np.array([0.75,-2.0,0.199599])}
                         _tid = self.config.train_reward_config.task_id
                         _g = _goals.get(_tid)
                         if _g is not None and len(trajs) > 0:
                             _reached = np.mean([1.0 if (np.max(t['rewards']) >= 1.0 - 1e-6) else 0.0 for t in trajs])
                             _final_d = np.mean([np.linalg.norm(np.asarray(t['observations'])[-1][19:22] - _g) for t in trajs])
                             _min_d = np.mean([np.min(np.linalg.norm(np.asarray(t['observations'])[:,19:22] - _g, axis=1)) for t in trajs])
                             print(f"[cube-diag] reached-goal-any-step: {_reached:.2f} | "
                                   f"mean final dist-to-goal: {_final_d:.3f} | mean closest dist: {_min_d:.3f}")
                 except Exception as _e:
                     print(f"[cube-diag] skipped ({_e})")
            self.accelerator.wait_for_everyone()

            if(self.config.offline):
                if(self.accelerator.is_main_process):
                     print(f"Finetuning round {step+1} completed")
                     print()
                self.accelerator.wait_for_everyone()
                continue



            if self.accelerator.is_main_process:
                  #print(f"Starting Reward Training")
                  if update_reward:
                      print(f"Starting Reward Training")
                      if(self.config.train_reward_config.ensemble_size is not None):
                          train_reward_ensemble(self.Train_Buffer,
                             dataset_name = self.config.dataset_name,
                             hidden_layers = self.config.train_reward_config.hidden_layers,
                             hidden_dim = self.config.train_reward_config.hidden_dim,
                             batch_size = self.config.train_reward_config.batch_size,
                             num_steps = self.config.train_reward_config.num_steps,
                             lr = self.config.train_reward_config.lr,
                             min_lr = self.config.train_reward_config.min_lr,
                             ensemble_size = self.config.train_reward_config.ensemble_size,
                             bootstrap = True,
                             save_percentage = 0.02,
                             sigma = self.config.train_reward_config.sigma,
                             step = ((step+1) * self.config.AMConfig.per_round_steps),
                             target_reward = self.config.train_reward_config.target_reward,
                             specific_dataset = self.config.specific_dataset,
                             goal = self.config.train_reward_config.train_goal,
                             task_id = self.config.train_reward_config.task_id)
                      else:

                         train_reward(self.Train_Buffer,
                             dataset_name = self.config.dataset_name,
                             hidden_layers = self.config.train_reward_config.hidden_layers,
                             hidden_dim = self.config.train_reward_config.hidden_dim,
                             batch_size = self.config.train_reward_config.batch_size,
                             num_steps = self.config.train_reward_config.num_steps,
                             lr = self.config.train_reward_config.lr,
                             min_lr = self.config.train_reward_config.min_lr,
                             sigma = self.config.train_reward_config.sigma,
                             step = ((step+1) * self.config.AMConfig.per_round_steps),
                             target_reward = self.config.train_reward_config.target_reward,
                             specific_dataset = self.config.specific_dataset,
                             goal = self.config.train_reward_config.train_goal,
                             task_id = self.config.train_reward_config.task_id)
                  """
                  if self.config.kernel:
                      print(f"Starting Kernel Training")
                      if(self.config.train_kernel_config.type_kernel == 'robust'):
                          threshold = train_kernel(self.Train_Kernel_Buffer,
                             dataset_name = self.config.dataset_name,
                             specific_dataset = self.config.specific_dataset,
                             batch_size = self.config.train_kernel_config.batch_size,
                             lr = self.config.train_kernel_config.lr,
                             num_steps = self.config.train_kernel_config.num_steps,
                             ensemble_size = self.config.train_kernel_config.ensemble_size,
                             λ_reg = self.config.train_kernel_config.λ_reg,
                             num_hidden_layers = self.config.train_kernel_config.num_hidden_layers,
                             hidden_dim = self.config.train_kernel_config.hidden_dim,
                             step = ((step+1) * self.config.AMConfig.per_round_steps),
                             constraint_type = self.config.RewardConfig.constraint_type,
                             quantile = self.config.RewardConfig.quantile,
                             x_generated_plans = x_generated_plans)

                      elif(self.config.train_kernel_config.type_kernel == 'mog'):
                          threshold = train_kernel_mog(self.Train_Kernel_Buffer,
                                      dataset_name = self.config.dataset_name,
                                      specific_dataset = self.config.specific_dataset,
                                      batch_size = self.config.train_kernel_config.batch_size,
                                      lr = self.config.train_kernel_config.lr,
                                      num_steps = self.config.train_kernel_config.num_steps,
                                      ensemble_size = self.config.train_kernel_config.ensemble_size,
                                      λ_reg = self.config.train_kernel_config.λ_reg,
                                      num_modes = self.config.train_kernel_config.kernel_num_modes,
                                      num_hidden_layers = self.config.train_kernel_config.num_hidden_layers,
                                      hidden_dim = self.config.train_kernel_config.hidden_dim,
                                      kernel_noise_floor = self.config.train_kernel_config.kernel_noise_floor,
                                      step = ((step+1) * self.config.AMConfig.per_round_steps),
                                      constraint_type = self.config.RewardConfig.constraint_type,
                                      quantile = self.config.RewardConfig.quantile,
                                      x_generated_plans = x_generated_plans)
                  """

                  if self.config.critic and self.config.update_critic and update_critic:
                      print(f"Starting Critic Training")
                      #save_trajs(critic_buffer, self.config.dataset_name, self.config.specific_dataset, ((step+1) * self.config.AMConfig.per_round_steps))
                      print(f"Number of trajectories of Critic Training: {len(critic_buffer)}")

                      train_critic(critic_buffer,
                                   dataset_name = self.config.dataset_name,
                                   specific_dataset = self.config.specific_dataset,
                                   hidden_layers = self.config.train_critic_config.hidden_layers,
                                   hidden_dim = self.config.train_critic_config.hidden_dim,
                                   sigma = self.config.train_reward_config.sigma,
                                   batch_size = self.config.train_critic_config.batch_size,
                                   num_steps = self.config.train_critic_config.num_steps,
                                   gamma = self.config.train_critic_config.gamma,
                                   lam = self.config.train_critic_config.lam,
                                   #horizon = self.config.AMConfig.horizon,
                                   #horizon = self.config.chunk_size,
                                   horizon = self.config.train_critic_config.horizon,
                                   lr = self.config.train_critic_config.lr,
                                   min_lr = self.config.train_critic_config.min_lr,
                                   tau = self.config.train_critic_config.tau,
                                   old_step = last_critic_update_step,
                                   new_step = ((step+1) * self.config.AMConfig.per_round_steps),
                                   momentum = self.config.train_critic_config.momentum,
                                   target_reward = self.config.train_reward_config.target_reward,
                                   task_id = self.config.train_reward_config.task_id)
            self.accelerator.wait_for_everyone()


            #plans = self.get_generated_plans(number_of_generated_plans = self.config.RewardConfig.number_of_generated_plans)

            if self.config.kernel and self.config.update_kernel:
                      plans = self.get_generated_plans(number_of_generated_plans = self.config.RewardConfig.number_of_generated_plans, rng = plan_key)
                      if self.accelerator.is_main_process:
                           print(f"Starting Kernel Training")
                      if(self.config.train_kernel_config.type_kernel == 'robust'):
                          threshold = train_kernel(self.Train_Kernel_Buffer,
                             dataset_name = self.config.dataset_name,
                             specific_dataset = self.config.specific_dataset,
                             batch_size = self.config.train_kernel_config.batch_size,
                             lr = self.config.train_kernel_config.lr,
                             num_steps = self.config.train_kernel_config.num_steps,
                             ensemble_size = self.config.train_kernel_config.ensemble_size,
                             λ_reg = self.config.train_kernel_config.λ_reg,
                             num_hidden_layers = self.config.train_kernel_config.num_hidden_layers,
                             hidden_dim = self.config.train_kernel_config.hidden_dim,
                             step = ((step+1) * self.config.AMConfig.per_round_steps),
                             constraint_type = "log_prob",
                             quantile = self.config.RewardConfig.quantile,
                             x_generated_plans = plans,
                             accelerator = self.accelerator)

                      elif(self.config.train_kernel_config.type_kernel == 'mog'):
                          threshold = train_kernel_mog(self.Train_Kernel_Buffer,
                                      dataset_name = self.config.dataset_name,
                                      specific_dataset = self.config.specific_dataset,
                                      batch_size = self.config.train_kernel_config.batch_size,
                                      lr = self.config.train_kernel_config.lr,
                                      num_steps = self.config.train_kernel_config.num_steps,
                                      ensemble_size = self.config.train_kernel_config.ensemble_size,
                                      λ_reg = self.config.train_kernel_config.λ_reg,
                                      num_modes = self.config.train_kernel_config.kernel_num_modes,
                                      num_hidden_layers = self.config.train_kernel_config.num_hidden_layers,
                                      hidden_dim = self.config.train_kernel_config.hidden_dim,
                                      kernel_noise_floor = self.config.train_kernel_config.kernel_noise_floor,
                                      step = ((step+1) * self.config.AMConfig.per_round_steps),
                                      constraint_type = "log_prob",
                                      quantile = self.config.RewardConfig.quantile,
                                      x_generated_plans = plans,
                                      accelerator = self.accelerator)

                      if threshold is not None:
                            if self.config.RewardConfig.min_log_prob > threshold:
                                self.config.RewardConfig.min_log_prob = threshold

            self.accelerator.wait_for_everyone()
            #set the new total reward model
            if update_reward:
                  self.config.reward_model_checkpoint = ((step+1) * self.config.AMConfig.per_round_steps)
                  last_reward_update_step = ((step+1) * self.config.AMConfig.per_round_steps)
            else:
                  self.config.reward_model_checkpoint = last_reward_update_step

            if(self.config.kernel and self.config.update_kernel):
                  self.config.kernel_model_checkpoint = ((step+1) * self.config.AMConfig.per_round_steps)
            else:
                  self.config.kernel_model_checkpoint = 0

            if(self.config.critic and self.config.update_critic):
                if (update_critic):
                     self.config.critic_model_checkpoint = ((step+1) * self.config.AMConfig.per_round_steps)
                     last_critic_update_step = ((step+1) * self.config.AMConfig.per_round_steps)
                else:
                     self.config.critic_model_checkpoint = last_critic_update_step
            else:
                 self.config.critic_model_checkpoint = 0



            self.set_reward_model(self.device)
            if self.accelerator.is_main_process:
                   print(f"Finetuning round {step+1} completed")
                   print()
            self.accelerator.wait_for_everyone()
