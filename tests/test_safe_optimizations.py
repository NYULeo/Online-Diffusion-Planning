import unittest

import numpy as np
import torch
from torch.autograd.functional import jvp

from Finetuning.traj_reward4 import _normalization_tensors
from Pretrain.utils import SAStats


class SafeOptimizationTest(unittest.TestCase):
    @staticmethod
    def freeze_module(module):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    def test_device_normalization_matches_existing_numpy_path(self):
        stats = SAStats()
        stats.obs_mean = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        stats.obs_std = np.array([2.0, 0.0, 4.0], dtype=np.float32)
        values = np.array([[3.0, 1.0, -1.5], [0.0, -2.0, 4.5]], dtype=np.float32)
        expected = stats.norm_obs(values)
        mean, std = _normalization_tensors(stats, torch.device("cpu"))
        actual = ((torch.from_numpy(values) - mean) / std).numpy()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_freezing_parameters_preserves_output_and_input_gradient(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1))
        value = torch.randn(4, 3)
        expected = model(value).detach()
        self.freeze_module(model)
        differentiable_value = value.clone().requires_grad_(True)
        actual = model(differentiable_value)
        gradient = torch.autograd.grad(actual.sum(), differentiable_value)[0]
        self.assertTrue(torch.equal(actual.detach(), expected))
        self.assertTrue(torch.isfinite(gradient).all())

    def test_jvp_does_not_require_parameter_gradients(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 4)
        value = torch.randn(2, 4)
        tangent = torch.randn_like(value)
        _, expected = jvp(model, (value,), (tangent,), create_graph=False)
        self.freeze_module(model)
        _, actual = jvp(model, (value,), (tangent,), create_graph=False)
        self.assertTrue(torch.equal(actual, expected))

    def test_disabling_higher_order_graph_preserves_first_gradient(self):
        value = torch.randn(8, requires_grad=True)
        expected = torch.autograd.grad(
            (value.sin() * value).sum(), value, create_graph=True
        )[0].detach()
        other = value.detach().clone().requires_grad_(True)
        actual = torch.autograd.grad(
            (other.sin() * other).sum(), other, create_graph=False
        )[0]
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
