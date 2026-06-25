'''Neural-network backbone base class for the Diffusion (score-matching) model — JAX/Flax port.'''
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .utils import SUPPORTED_TIMESTEP_EMBEDDING


class BaseNNDiffusion(nn.Module):
    """
    The neural network backbone for the Diffusion model used for score matching
     (or training a noise predictor) should take in three inputs.
     The first input is the noisy data.
     The second input is the denoising time step, which can be either as a discrete variable
     or a continuous variable, specified by the parameter `discrete_t`.
     The third input is the condition embedding that has been processed through the `nn_condition`.
     In the general case, we assume that there may be multiple conditions,
     which are inputted as a tensor dictionary, or a single condition, directly inputted as a tensor.
    """

    emb_dim: int
    timestep_emb_type: str = 'positional'
    timestep_emb_params: Optional[dict] = None

    def setup(self):
        assert self.timestep_emb_type in SUPPORTED_TIMESTEP_EMBEDDING.keys()
        timestep_emb_params = self.timestep_emb_params or {}
        self.map_noise = SUPPORTED_TIMESTEP_EMBEDDING[self.timestep_emb_type](self.emb_dim, **timestep_emb_params)

    def __call__(self,
                 x: jnp.ndarray, noise: jnp.ndarray,
                 condition: Optional[jnp.ndarray] = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        raise NotImplementedError
