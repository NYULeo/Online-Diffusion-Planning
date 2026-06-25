'''Critic training / testing for ODP (JAX/Flax port of the PyTorch originals, FQL-style).'''

from pathlib import Path
import copy
import pickle
from typing import Optional, List

import numpy as np
from sympy.integrals.meijerint import _rewrite_single
from scipy.ndimage import gaussian_filter1d

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax

from Finetuning.utils import TrajectoryDict, get_trajs, getName, check_device
from Pretrain.Dataset import get_dataset, get_env
from Pretrain.utils import set_seed, SAStats, cycle, ema_smooth
from Pretrain.Critic.nets import Critic, CriticEnsemble

# Shared port plumbing (mirrors fql).
from JAX_PORT.jax_utils import (
    MLP, ModuleDict, TrainState, nonpytree_field, default_init, ensemblize,
    target_update, save_agent, restore_agent, supply_rng,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]   # Online-Diffusion-Planning/
PRETRAIN_DIR = PROJECT_ROOT / "Pretrain"
FINETUNE_DIR = PROJECT_ROOT / "Finetuning"


class RunningMeanStd:
    """
    Maintains running mean and standard deviation for online normalization.
    Very useful for stabilizing value function training.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.std = 1.0
        self.count = 0
        self.epsilon = epsilon

    def update(self, x):
        """
        Update statistics with new batch of data.
        x can be a jax/numpy array.
        """
        x = np.asarray(x).flatten()
        batch_count = len(x)

        if batch_count == 0:
            return

        batch_mean = np.mean(x)
        batch_var = np.var(x)

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            self.mean = (self.count * self.mean + batch_count * batch_mean) / total_count

            self.var = (self.count * self.var + batch_count * batch_var +
                       (self.count * batch_count * delta ** 2) / total_count) / total_count

        self.std = np.sqrt(self.var + self.epsilon)
        self.count += batch_count

    def normalize(self, x):
        """Normalize input array"""
        return (np.asarray(x) - self.mean) / (self.std + self.epsilon)

    def denormalize(self, x):
        """Denormalize (useful when using the value for policy)"""
        return np.asarray(x) * (self.std + self.epsilon) + self.mean


def spare_reward_prcocessor(rewards):
    Temp = []
    for i in range(1, len(rewards)):
        if(rewards[i] == rewards[i-1]+1):
            Temp.append(i)
    new_rewards = [0]*len(rewards)
    for i in range(len(rewards)):
        if(i in Temp):
            new_rewards[i] = 1.0
        else:
            new_rewards[i] = 0.0
    return np.array(new_rewards, dtype = np.float64)
    #return np.array(new_rewards)

def save_critic_hyperparameters(dataset_name, batch_size, num_steps, lr, min_lr, sigma, alpha,
                                obs_dim, critic_net, optimizer, gamma, horizon, tau,
                                specific_dataset: Optional[str] = None,
                                target_reward: Optional[float] = None,
                                goal: Optional[np.array] = None, task_id: Optional[int] = None):

    """
    os.makedirs(f"./Pretrain/Critic/{dataset_name}/{specific_dataset}/args/", exist_ok=True)
    filepath = f"./Pretrain/Critic/{dataset_name}/{specific_dataset}/args/hyperparameters.json"
    """
    args_dir = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "args"
    args_dir.mkdir(parents=True, exist_ok=True)
    filepath = args_dir / "hyperparameters.json"

    def convert_to_json_serializable(obj):
        """Recursively convert objects to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)
        return obj

    # Get optimizer info
    optimizer_type = type(optimizer).__name__
    optimizer_params = {
        'type': optimizer_type,
        'lr': lr,
        # optax optimizers carry no per-group weight_decay introspection; ODP critic uses Adam (wd=0).
        'weight_decay': 0,
    }

    # Get model architecture info
    model_info = {
        'model_type': type(critic_net).__name__,
        'obs_dim': int(obs_dim),
    }

    # Add model-specific parameters if available
    if hasattr(critic_net, 'hidden'):
        model_info['hidden_dim'] = int(critic_net.hidden)

    # Compile all hyperparameters
    hyperparams = {
        'env_details': {
            'dataset_name': dataset_name,
            'specific_dataset': specific_dataset,
            'obs_dim': int(obs_dim),
        },
        'model_architecture': model_info,
        'training_hyperparameters': {
            'num_steps': num_steps,
            'batch_size': batch_size,
            'lr': lr,
            'min_lr': min_lr,
            'optimizer': optimizer_params,
        },
        'critic_config': {
            'gamma': float(gamma),
            'horizon': int(horizon),
            'tau': float(tau),
        },
        'reward_processing': {
            'sigma': float(sigma) if sigma is not None else None,
            'alpha': float(alpha) if alpha is not None else None,
            'target_reward': target_reward,
            'goal': convert_to_json_serializable(goal),
            'task_id': task_id
        }
    }

    # Handle numpy arrays and other non-JSON-serializable types
    hyperparams = convert_to_json_serializable(hyperparams)

    # Save with pretty printing (indent=4 makes it human-readable)
    import json
    with open(filepath, 'w') as f:
        json.dump(hyperparams, f, indent=4, sort_keys=False)

    print(f"Critic pretraining hyperparameters saved to {filepath}", flush=True)

def get_CriticName(env_name, specific_env, task_id: Optional[int] = None):
     if(env_name == 'kitchen'):
          if(specific_env == 'complete'):
               return 'Kitchen_High'
          elif(specific_env == 'partial'):
               return 'Kitchen_Medium'
          elif(specific_env == 'mixed'):
               return 'Kitchen_Mixed'
          else:
               raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'pointmaze'):
         if(specific_env == 'large'):
              return 'PointMaze_Large'
         elif(specific_env == 'medium'):
              return 'PointMaze_Medium'
         elif(specific_env == 'unmaze'):
              return 'PointMaze_Unmaze'
         else:
              raise ValueError(f"Invalid specific environment: {specific_env}")
     elif(env_name == 'cube'):
         if specific_env == 'single-play':
              return f'Cube_SinglePlay_task{task_id}'
         elif specific_env == 'single-noisy':
             return f'Cube_SingleNoisy_task{task_id}'
         elif specific_env == 'double-play':
             return f'Cube_DoublePlay_task{task_id}'
         elif specific_env == 'double-noisy':
             return f'Cube_DoubleNoisy_task{task_id}'
         elif specific_env == 'triple-play':
             return f'Cube_TriplePlay_task{task_id}'
         elif specific_env == 'triple-noisy':
             return f'Cube_TripleNoisy_task{task_id}'
         elif specific_env == 'quadruple-play':
             return f'Cube_QuadruplePlay_task{task_id}'
         elif specific_env == 'quadruple-noisy':
             return f'Cube_QuadrupleNoisy_task{task_id}'
         else:
             raise ValueError(f"Invalid cube dataset name: {specific_env}")
     else:
         raise ValueError(f"Invalid environment name: {env_name}")

def save_critic(model, dataset_name, specific_dataset, step, task_id: Optional[int] = None):
    # `model` is a TrainState (the JAX critic). Serialize its params (flax.serialization).
    name = get_CriticName(dataset_name, specific_dataset, task_id)
    net_dict = flax.serialization.to_state_dict(model.params)
    models_dir = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f"{name}_Critic_{step}.pkl"
    #print("Exists:", os.path.isfile(save_path), "Size:", os.path.getsize(save_path) if os.path.isfile(save_path) else None)
    # TODO(checkpoint-bridge): torch original did `torch.save(model.state_dict(), save_path)`.
    # The flax param tree differs in key/layout from the torch state_dict; consumers loading these
    # checkpoints (get_critic_model) must use the matching flax loader, not torch.load.
    with open(save_path, "wb") as f:
        pickle.dump(net_dict, f)
    print(f"critic model save to {name}.pkl")

def save_to_finetuning(critic_net, dataset_name, specific_dataset, task_id: Optional[int] = None):
    # `critic_net` is a TrainState (the JAX critic). Serialize its params (flax.serialization).
    net_dict = flax.serialization.to_state_dict(critic_net.params)
    name = get_CriticName(dataset_name, specific_dataset, task_id)
    ft_models_dir = FINETUNE_DIR / "Critics" / dataset_name / specific_dataset / "Models"
    ft_models_dir.mkdir(parents=True, exist_ok=True)
    save_path = ft_models_dir / f"{name}_Critic_0.pkl"
    # TODO(checkpoint-bridge): torch original did `torch.save(critic_net.state_dict(), save_path)`.
    with open(save_path, "wb") as f:
        pickle.dump(net_dict, f)
    print(f"critic model save to {save_path}")

def save_stats_to_finetuning(stats, dataset_name, specific_dataset: Optional[str] = None, task_id: Optional[int] = None):
    name = get_CriticName(dataset_name, specific_dataset, task_id)
    ft_stats_dir = FINETUNE_DIR / "Critics" / dataset_name / specific_dataset / "Stats"
    ft_stats_dir.mkdir(parents=True, exist_ok=True)
    savepath = ft_stats_dir / f"{name}_Critic_stats_0.pkl"
    with open(savepath, "wb") as f:
        pickle.dump(stats, f)
    print(f"saved stats to {savepath}")

def get_critic_model(dataset_name, specific_dataset, step, task_id: Optional[int] = None):
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)
    name = get_CriticName(dataset_name, specific_dataset, task_id)
    path = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "Models" / f"{name}_Critic_{step}.pkl"
    # TODO(checkpoint-bridge): torch original did
    #   model_state_dict = torch.load(path, weights_only=True, map_location="cpu")
    # We now load the flax-serialized param state_dict pickled by save_critic. If a *legacy torch*
    # checkpoint is ingested here, a torch->flax key remap (Linear weight.T -> kernel, LayerNorm
    # weight -> scale, etc.; see CONVERSION_GUIDE §10) must be inserted before returning.
    with open(path, "rb") as f:
        model_state_dict = pickle.load(f)
    return model_state_dict, obs_dim

def get_critic_stats(dataset_name, specific_dataset, task_id: Optional[int] = None):
    name = get_CriticName(dataset_name, specific_dataset, task_id)
    path = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "Stats" / f"{name}_Critic_stats.pkl"
    with open(path, "rb") as f:
        stats = pickle.load(f)
    return stats

class CriticDataset:
    def __init__(self, dataset_name: str, specific_dataset: str, trajs: List[TrajectoryDict], target_reward: Optional[float] = None, horizon: int = 32, gamma: float = 0.99, sigma: float = 7.0, alpha: Optional[float] = None, task_id: Optional[int] = None):

        obs_all = []
        for traj in trajs:
            obs_all.append(traj['observations'])
        obs_all = np.concatenate(obs_all, axis = 0)

        #get stats
        self.stats = SAStats()
        self.stats.obs_mean = obs_all.mean(axis=0)
        self.stats.obs_std = obs_all.std(axis=0)+ 1e-8
        allowed_values = [0.0, 1.0]

        transitions = []
        for traj in trajs:
            obs = traj['observations']
            rews = traj['rewards']
            #rews = spare_reward_prcocessor(rews)
            if(not np.all(np.isin(rews, allowed_values))):
                raise ValueError(f"Rewards must be etiher 0 or 1, but got {rews}")

            if(target_reward is not None):
                rews = self.boost_signal(target_reward, rews)
            if(alpha is not None):
                rews = ema_smooth(rews, alpha)
            elif(sigma is not None):
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate = 200/sigma)
            #if(len(obs) > horizon):
            rews = self.reward_processor(rews, horizon, gamma)
            #for t in range(len(obs)-horizon+1):
            for t in range(len(rews)):
                     obs_t = self.stats.norm_obs(obs[t])
                     r_t   = rews[t]
                     obs_next_t = self.stats.norm_obs(obs[min(t + horizon, len(obs) - 1)])
                     if(t + horizon < len(obs)):
                        done = False
                     else:
                        done = True
                     transitions.append((obs_t, r_t, obs_next_t, done))


        self.transitions = transitions
        self.save_stats(dataset_name, specific_dataset, task_id)


    def save_stats(self, dataset_name, specific_dataset, task_id: Optional[int] = None):
        name = get_CriticName(dataset_name, specific_dataset, task_id)
        stats_name =  str(name) + f'_Critic_stats.pkl'
        stats_dir = PRETRAIN_DIR / "Critic" / dataset_name / specific_dataset / "Stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        savepath = stats_dir / stats_name
        with open(savepath, "wb") as f:
            pickle.dump(self.stats, f)
        print(f"saved stats to {savepath}")

    def __len__(self):
        return len(self.transitions)#

    def __getitem__(self, idx):
        s, r, s_next, done = self.transitions[idx]
        return (
                np.asarray(s, dtype=np.float32),
                np.asarray(r, dtype=np.float32),
                np.asarray(s_next, dtype=np.float32),
                np.asarray(done, dtype=np.float32),
            )

    def sample(self, batch_size):
        """fql-style host-side batch sampler (replaces torch DataLoader iteration).

        Returns a tuple `(s, r, s_next, done)` of stacked numpy arrays, matching the per-item layout
        produced by `__getitem__`. Random index draw uses numpy RNG (data shuffling stays host-side
        per CONVERSION_GUIDE §13).
        """
        idxs = np.random.randint(0, len(self.transitions), size=batch_size)
        s, r, s_next, done = [], [], [], []
        for idx in idxs:
            si, ri, sni, di = self[idx]
            s.append(si)
            r.append(ri)
            s_next.append(sni)
            done.append(di)
        return (
            np.stack(s, axis=0),
            np.stack(r, axis=0),
            np.stack(s_next, axis=0),
            np.stack(done, axis=0),
        )

    def boost_signal(self, target_reward, rews):
        rews = np.asarray(rews, dtype=np.float64).copy()
        rews = rews * target_reward
        return rews

    def reward_processor(self, rews, horizon, gamma):
        new_rews = []
        for t in range(len(rews)):
            R = 0.0
            for i in range(t, min(t + horizon, len(rews))):
                R += (gamma**(i-t))*rews[i]
                #R += (gamma**(i-t+1))*rews[i]
            new_rews.append(R)
            #new_rews.append(np.sum(rews[t:]))
        return new_rews

def train_critic(dataset_name: str, specific_dataset: str, hidden_layers: int, hidden_dim: int, batch_size, num_steps, gamma, horizon, lr, min_lr, tau, sigma: Optional[float] = None, alpha: Optional[float] = None, target_reward = 1.0, trajs: List[TrajectoryDict] = None, task_id: Optional[int] = None, *, rng=None):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device threading is a no-op in JAX (CONVERSION_GUIDE §9); check_device kept importable but unused.

    if rng is None:
        rng = jax.random.PRNGKey(0)

    dataset = CriticDataset(dataset_name, specific_dataset, trajs, target_reward, horizon, gamma, sigma, alpha, task_id)
    _, obs_dim, _ = get_env(dataset_name, specific_dataset)


    dataloader = cycle(iter(lambda: dataset.sample(batch_size), None))
    critic_def = Critic(obs_dim, hidden_dim, hidden_layers)
    # Cosine LR schedule (one step per training step), eta_min = min_lr. Matches torch
    # CosineAnnealingLR(T_max=num_steps, eta_min=min_lr) folded into the optax optimizer.
    schedule = optax.cosine_decay_schedule(lr, num_steps, alpha=min_lr / lr)
    # grad clip (max_norm=1.0) chained before Adam, matching torch clip_grad_norm_ + Adam.
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=schedule))

    # Build params + TrainState for the online critic; target_critic mirrors the same init (load_state_dict).
    rng, init_rng = jax.random.split(rng)
    example_s = np.zeros((1, int(obs_dim)), dtype=np.float32)
    params = critic_def.init(init_rng, example_s)['params']
    critic = TrainState.create(critic_def, params, tx=tx)
    target_critic = TrainState.create(critic_def, jax.tree_util.tree_map(lambda x: x, params))

    print(f"Training critic for {dataset_name}-{specific_dataset}")
    #return_rms = RunningMeanStd(epsilon = 1e-8)

    @jax.jit
    def update_step(critic, target_critic, s, r, s_next, done):
        # Compute target V-values (no grad through the target net; call without params=).
        q_next = target_critic(s_next)
        target = r + ((gamma ** horizon) * q_next * (1 - done.astype(jnp.float32)))

        def loss_fn(params):
            # Predicted V-values
            q_pred = critic(s, params=params)
            # Original computed an MSE then overwrote it with smooth_l1 (beta=1.0); keep smooth_l1.
            diff = jnp.abs(q_pred - target)
            loss = jnp.mean(jnp.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))
            return loss, {'loss': loss}

        new_critic, info = critic.apply_loss_fn(loss_fn=loss_fn)
        # Soft update target network: tgt = (1 - tau) * tgt + tau * online (target_update(p, tp, tau)).
        new_target_params = target_update(new_critic.params, target_critic.params, tau)
        new_target_critic = target_critic.replace(params=new_target_params)
        return new_critic, new_target_critic, info['loss']

    total_loss = 0.0
    for k in range(1, num_steps + 1):  # number of passes over dataset
           s, r, s_next, done = next(dataloader)   # r is now n-step return
           s = jnp.asarray(s)
           r = jnp.asarray(r)
           s_next = jnp.asarray(s_next)
           done = jnp.asarray(done)

           critic, target_critic, loss = update_step(critic, target_critic, s, r, s_next, done)
           total_loss += float(loss)

           if(k % 2000 == 0):
                avg_loss = total_loss/2000
                print(f"Average Loss: {avg_loss:.4f}")
                try:
                    from wandb_logger import wlog
                    wlog({'critic/loss': avg_loss, 'critic/lr': float(schedule(critic.step - 1))}, step=k)
                except Exception:
                    pass
                total_loss = 0.0

           if(k % 10000 == 0):
                save_critic(target_critic, dataset_name, specific_dataset, k, task_id)
                print(f"Checkpoint saved at step {k}")
    save_to_finetuning(target_critic, dataset_name, specific_dataset, task_id)
    stats = get_critic_stats(dataset_name, specific_dataset, task_id)
    save_stats_to_finetuning(stats, dataset_name, specific_dataset, task_id)
    print(f"critic model saved")

class Critic_Test_Dataset:
    def __init__(self,
                 dataset_name: str,
                 specific_dataset: str,
                 trajs: List[TrajectoryDict],
                 sigma: Optional[float] = None,
                 task_id: Optional[int] = None,
                 target_reward: Optional[float] = None,
                 horizon: int = 32,
                 gamma: float = 0.99):

        self.stats = get_critic_stats(dataset_name, specific_dataset, task_id)
        self.horizon = horizon
        self.gamma = gamma

        transitions = []
        for traj in trajs:
            obs = traj['observations']
            rews = traj['rewards'].copy()

            if not np.all(np.isin(rews, [0.0, 1.0])):
                raise ValueError(f"Rewards must be either 0 or 1, but got {rews}")

            if target_reward is not None:
                rews = self.boost_signal(target_reward, rews)
            if sigma is not None:
                rews = gaussian_filter1d(rews, sigma, mode="nearest", truncate=200/sigma)

            for t in range(len(obs) - horizon):        # consistent with training
                obs_t = self.stats.norm_obs(obs[t])
                rews_chunk = rews[t : t + horizon]
                transitions.append((obs_t, rews_chunk))

        self.transitions = transitions
        print(f"Test dataset created: {len(self.transitions)} samples (horizon={horizon})")

    def boost_signal(self, target_reward, rews):
        rews = rews.copy()
        rews[rews == 1.0] = target_reward
        return rews

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        obs_t, rews_chunk = self.transitions[idx]
        return (
            np.asarray(obs_t, dtype=np.float32),
            np.asarray(rews_chunk, dtype=np.float32),
        )

    def batches(self, batch_size):
        """fql-style sequential batching (replaces DataLoader(shuffle=False, drop_last=False)).

        Yields `(s, rews_chunk)` numpy arrays in dataset order, matching the torch eval loop.
        """
        n = len(self.transitions)
        for start in range(0, n, batch_size):
            idxs = range(start, min(start + batch_size, n))
            s, rc = [], []
            for idx in idxs:
                si, rci = self[idx]
                s.append(si)
                rc.append(rci)
            yield np.stack(s, axis=0), np.stack(rc, axis=0)

def test_critic(dataset_name: str,
                specific_dataset: str,
                hidden_layers: int,
                hidden_dim: int,
                checkpoint_step: int,
                gamma: float = 0.99,
                horizon: int = 32,
                sigma: Optional[float] = None,
                target_reward: float = 1.0,
                trajs: List[TrajectoryDict] = None,
                task_id: Optional[int] = None):

    # device threading is a no-op in JAX (CONVERSION_GUIDE §9).

    dataset = Critic_Test_Dataset(
        dataset_name, specific_dataset, trajs,
        sigma, task_id, target_reward, horizon, gamma
    )

    # Load model (preserves the torch call's argument order, including its existing task_id/step swap).
    model_state_dict, obs_dim = get_critic_model(dataset_name, specific_dataset, task_id, checkpoint_step)
    model_def = Critic(obs_dim, hidden_dim, hidden_layers)
    # TODO(checkpoint-bridge): torch original did model.load_state_dict(model_state_dict).
    # model_state_dict is the flax-serialized param tree saved by save_critic; restore it into a fresh
    # param template so `model_def.apply` can run it.
    template_params = model_def.init(jax.random.PRNGKey(0), np.zeros((1, int(obs_dim)), dtype=np.float32))['params']
    params = flax.serialization.from_state_dict(template_params, model_state_dict)
    model = TrainState.create(model_def, params)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    print(f"Testing critic at checkpoint {checkpoint_step} (consistent with training)...")

    for s, rews_chunk in dataset.batches(100):
        s = jnp.asarray(s)
        rews_chunk = jnp.asarray(rews_chunk)          # (B, horizon)

        pred = model(s)                               # V(s) - shape (B, 1) or (B,)

        if pred.ndim == 2:
            pred = jnp.squeeze(pred, 1)

        # Compute same style target as training: n-step return
        target = jnp.zeros_like(pred)
        for i in range(rews_chunk.shape[1]):
            target = target + (gamma ** i) * rews_chunk[:, i]

        diff = jnp.abs(pred - target)
        loss = jnp.mean(jnp.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))
        total_loss += float(loss) * s.shape[0]

        all_preds.extend(np.asarray(pred))
        all_targets.extend(np.asarray(target))

    avg_loss = total_loss / len(dataset)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))

    print(f"Test Results (Checkpoint {checkpoint_step}):")
    print(f"   Smooth L1 Loss : {avg_loss:.4f}")
    print(f"   MAE            : {mae:.4f}")
    print(f"   Mean Pred      : {np.mean(all_preds):.3f}")
    print(f"   Mean Target    : {np.mean(all_targets):.3f}")
    print(f"   Pred Std       : {np.std(all_preds):.3f}")

    return avg_loss, mae

