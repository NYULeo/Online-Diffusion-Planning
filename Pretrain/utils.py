'''Miscellaneous utilities for the ODP pretrain pipeline (seeding, normalization stats, device).'''
from typing import Optional

import jax
import jax.numpy as jnp
import flax
import numpy as np
import math
import random
import os
import pickle


def cycle(dl):
    while True:
        for data in dl:
            yield data


class SAStats:
    obs_mean: np.ndarray
    obs_std:  np.ndarray
    #act_min =  np.array([-1.0] * 9)
    #act_max =  np.array([ 1.0] * 9)
    eps: float = 1e-3
    std_floor: float = 1e-3

    # ---- observation ----
    def norm_obs(self, s: np.ndarray) -> np.ndarray:
        std = np.maximum(self.obs_std, self.std_floor)
        return (s - self.obs_mean) / (std)

    def denorm_obs(self, s: np.ndarray) -> np.ndarray:
        std = np.maximum(self.obs_std, self.std_floor)
        return s * (std) + self.obs_mean


def set_seed(seed=0):
    # Python random
    random.seed(seed)
    # NumPy random
    np.random.seed(seed)
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    # JAX has no global RNG; return a key for the caller to thread (see CONVERSION_GUIDE §8).
    return jax.random.PRNGKey(seed)


def compare_models_state_dict(model1, model2, tolerance=1e-6):
    """
    Compare two models by their state dictionaries.
    Returns True if models are identical within tolerance.
    """
    # Get state dictionaries (flax flattened state dicts of the param pytrees).
    state_dict1 = flax.traverse_util.flatten_dict(flax.serialization.to_state_dict(model1), sep='/')
    state_dict2 = flax.traverse_util.flatten_dict(flax.serialization.to_state_dict(model2), sep='/')

    # Check if they have the same keys
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        print("Models have different parameter names")
        return False

    # Compare each parameter
    for key in state_dict1.keys():
        param1 = jnp.asarray(state_dict1[key])
        param2 = jnp.asarray(state_dict2[key])

        # Check shapes
        if param1.shape != param2.shape:
            print(f"Parameter {key} has different shapes: {param1.shape} vs {param2.shape}")
            return False

        # Check values
        if not jnp.allclose(param1, param2, atol=tolerance):
            print(f"Parameter {key} has different values (max diff: {jnp.max(jnp.abs(param1 - param2))})")
            return False

    print("Models are identical!")
    return True




def ema_smooth(rewards, alpha = 0.99):

    rewards = np.asarray(rewards)
    assert rewards.ndim == 1, "rewards must be 1D (length T)"
    assert 0.0 < alpha < 1.0, "alpha must be in (0, 1)"

    beta = 1.0 - alpha
    rewards_smooth = np.zeros_like(rewards)

    # Initialize EMA with the first reward
    rewards_smooth[0] = rewards[0]
    for t in range(1, len(rewards)):
        rewards_smooth[t] = alpha * rewards_smooth[t - 1] + beta * rewards[t]

    return rewards_smooth



def check_device():
    device = jax.default_backend()
    if device == 'gpu':
        print("✅ Using GPU backend")
    elif device == 'tpu':
        print("✅ Using TPU backend")
    else:
        print("⚠️  Falling back to CPU (no GPU acceleration)")
    return device
