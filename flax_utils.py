"""Shared JAX/Flax helpers for the ODP port.

This module mirrors the idioms of the reference repo `fql` (Flow Q-Learning by Seohong Park),
specifically `utils/flax_utils.py` and the generic helpers in `utils/networks.py`. Every converted
ODP file should import its framework plumbing from here so that all ~30 parallel conversions share a
single, identical set of primitives.

Import-clean: depends only on jax / flax / optax / numpy.
Style: single quotes, 120 columns.
"""

import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

# ----------------------------------------------------------------------------------------------------------------------
# Struct field that is NOT a pytree leaf (use for config dicts, optimizers, static metadata on PyTreeNodes).
# ----------------------------------------------------------------------------------------------------------------------
nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


# ----------------------------------------------------------------------------------------------------------------------
# Network initialization / generic linen modules (mirrors fql/utils/networks.py).
# ----------------------------------------------------------------------------------------------------------------------
def default_init(scale=1.0):
    """Default kernel initializer (variance scaling, fan_avg, uniform).

    This matches fql's default. NOTE: this is NOT identical to PyTorch's default `nn.Linear` init
    (which is Kaiming-uniform on fan_in). When numerics must match a torch checkpoint, pass an
    explicit initializer instead (see CONVERSION_GUIDE.md, "Parameter init faithfulness").
    """
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module by vmapping over a leading `num_qs` axis of its params.

    Mirrors fql. Use for critic/reward/kernel ensembles where torch stacked weights or a python list
    of modules. `split_rngs={'params': True}` gives each ensemble member independent initialization.
    """
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron (verbatim from fql/utils/networks.py).

    Attributes:
        hidden_dims: Hidden layer dimensions (the last entry is the output dim).
        activations: Activation function applied between layers.
        activate_final: Whether to apply the activation (and optional LayerNorm) to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization after each activation.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# ModuleDict: bundle several named submodules into one set of params (verbatim from fql/utils/flax_utils.py).
# ----------------------------------------------------------------------------------------------------------------------
class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# TrainState: holds model_def, params, optax tx, opt_state (verbatim from fql/utils/flax_utils.py).
# ----------------------------------------------------------------------------------------------------------------------
class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        `loss_fn` must take `params` and return `(loss, info_dict)` (i.e. `has_aux=True`).
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


# ----------------------------------------------------------------------------------------------------------------------
# Target / EMA network update (generalized from fql's per-agent `target_update`).
# ----------------------------------------------------------------------------------------------------------------------
def target_update(params, target_params, tau):
    """Polyak (EMA) update of a target parameter pytree.

    new_target = tau * params + (1 - tau) * target_params

    Mirrors fql's FQLAgent.target_update convention (`p * tau + tp * (1 - tau)`). Use anywhere torch
    code maintained a target/EMA copy of a network (e.g. critic target nets, planner EMA in
    `EMA.update_model_average`). `tau` is the interpolation weight on the *online* params.
    """
    return jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), params, target_params)


# ----------------------------------------------------------------------------------------------------------------------
# RNG convenience helpers (explicit PRNG threading; no global RNG).
# ----------------------------------------------------------------------------------------------------------------------
def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Wrap a function that takes a `seed=` keyword so successive calls auto-thread a fresh key.

    Mirrors fql/utils/evaluation.py's `supply_rng`. Useful for evaluation/rollout policy callables
    that torch wrote as stateful samplers.
    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def split(rng, num=2):
    """Thin alias for `jax.random.split` (kept for call-site readability when threading keys)."""
    return jax.random.split(rng, num)


# ----------------------------------------------------------------------------------------------------------------------
# Save / restore (verbatim from fql/utils/flax_utils.py; flax.serialization-based).
# ----------------------------------------------------------------------------------------------------------------------
def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent (any flax.struct.PyTreeNode / TrainState-bearing object).
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent template (structure to restore into).
        restore_path: Glob path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
