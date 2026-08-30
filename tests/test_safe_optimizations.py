import unittest

import numpy as np
import torch
from torch.autograd.functional import jvp
import types

from Finetuning.acc_adjoint_matching import Acc_AdjointMatchingFineTuner
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

    def test_batched_sampler_preserves_scalar_rng_order(self):
        class Score(torch.nn.Module):
            def forward(self, value, timestep):
                return 0.1 * value + timestep[:, None, None]

        class Reward:
            def predict(self, plan, lam):
                return plan.square().mean() + lam

        for eta in (0.0, 0.7):
            tuner = Acc_AdjointMatchingFineTuner.__new__(Acc_AdjointMatchingFineTuner)
            tuner.device = torch.device("cpu")
            tuner.new_score_net = Score()
            tuner.config = types.SimpleNamespace(
                batch_per_sample=2, d_s=2, d_a=1, horizon=4,
                diffusion_steps=3, num_karras=1, eta=eta,
            )
            tuner.sigma_grid = torch.tensor([1.0, 0.7, 0.3, 0.0])
            tuner.t_grid = torch.tensor([1.0, 0.66, 0.33, 0.0])
            tuner.beta_1 = torch.tensor([0.8, 0.7, 0.6, 0.5])
            tuner.beta_2 = torch.tensor([0.4, 0.3, 0.2, 0.1])
            tuner.Lam = types.SimpleNamespace(get_lam=lambda: 0.25)
            states = torch.tensor([[0.1, 0.2], [-0.3, 0.4]])

            torch.manual_seed(123)
            scalar_trajectories = []
            scalar_rewards = []
            for state in states:
                for _ in range(tuner.config.batch_per_sample):
                    trajectory, reward = tuner.sample_Traj_karras(state, Reward())
                    scalar_trajectories.append(trajectory)
                    scalar_rewards.append(reward)
            scalar_trajectories = torch.stack(scalar_trajectories)
            scalar_rewards = torch.stack(scalar_rewards)

            torch.manual_seed(123)
            batch_trajectories, batch_rewards = tuner.sample_trajs_karras_batch(
                states, Reward()
            )
            self.assertTrue(torch.equal(batch_trajectories, scalar_trajectories))
            self.assertTrue(torch.equal(batch_rewards, scalar_rewards))

    def test_vectorized_adjoint_loss_matches_scalar_formula(self):
        class Score(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(scale))

            def forward(self, value, timestep):
                return self.scale * value + timestep[:, None, None]

        tuner = Acc_AdjointMatchingFineTuner.__new__(Acc_AdjointMatchingFineTuner)
        tuner.device = torch.device("cpu")
        tuner.config = types.SimpleNamespace(
            num_Loss_Clip_steps=1, reward_scaling_factor=3.0
        )
        tuner.t_asc = torch.linspace(0.9, 0.1, 4)
        tuner.k = -torch.tensor([0.3, 0.5, 0.7, 0.9])
        tuner.new_score_net = Score(0.2)
        tuner.old_score_net = Score(-0.1)
        trajectory = [torch.randn(1, 3, 2) for _ in range(4)]
        adjoints = [torch.randn(1, 3, 2) for _ in range(4)]
        vectorized = tuner.adjoint_matching_loss(trajectory, adjoints)

        scalar = torch.tensor(0.0)
        for index, (state, adjoint) in enumerate(zip(trajectory, adjoints)):
            k_value = tuner.k[index]
            timestep = tuner.t_asc[index : index + 1]
            new_v = k_value * state + k_value * tuner.new_score_net(state, timestep)
            old_v = k_value * state + k_value * tuner.old_score_net(state, timestep)
            sigma = torch.sqrt(-2 * k_value)
            value = ((new_v - old_v) * (2 / sigma) + sigma * adjoint).square().mean()
            if index <= tuner.config.num_Loss_Clip_steps:
                value = torch.minimum(value, torch.tensor(14.4))
            scalar += value
        self.assertTrue(torch.allclose(vectorized, scalar / 4, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
