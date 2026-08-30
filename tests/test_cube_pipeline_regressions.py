import importlib.machinery
import sys
import types
import unittest

import numpy as np
import torch


wandb = types.ModuleType("wandb")
wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
wandb.init = wandb.log = wandb.finish = lambda *args, **kwargs: None
sys.modules.setdefault("wandb", wandb)

from Finetuning.acc_adjoint_matching import Acc_AdjointMatchingFineTuner
from Finetuning.traj_reward4 import TotalReward_Critic
from Finetuning.utils import CriticDataset_Reward, Critic_Buffer_Reward
from Pretrain.Critic.nets import Critic
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras_batch


class ZeroScore(torch.nn.Module):
    def forward(self, value, timestep):
        if timestep.shape != (value.shape[0],):
            raise AssertionError((value.shape, timestep.shape))
        return torch.zeros_like(value)


class CubePipelineRegressionTest(unittest.TestCase):
    def test_batched_sampler_conditions_all_candidates(self):
        state = np.array([0.25, -0.5, 1.0], dtype=np.float32)
        plans = sample_euler_karras_batch(
            state, ZeroScore(), 3, 2, 5,
            num_steps=3, num_karras=1, eta=0.0,
            device="cpu", num_samples=4,
        )
        self.assertEqual(plans.shape, (4, 5, 5))
        np.testing.assert_allclose(
            plans[:, 0, :3], np.repeat(state[None], 4, axis=0)
        )

    def test_critic_is_nonnegative(self):
        critic = Critic(3, hidden_dim=16, hidden_layers=1)
        self.assertTrue(torch.all(critic(torch.randn(64, 3)) >= 0))

    def test_short_trajectory_keeps_terminal_reward(self):
        dataset = CriticDataset_Reward.__new__(CriticDataset_Reward)
        dataset.horizon = 4
        dataset._normalized_observations = [
            np.array([[0.0], [1.0]], dtype=np.float32)
        ]
        dataset._scaled_rewards = [np.array([0.0, 2.0], dtype=np.float32)]
        dataset._windows = [(0, 0), (0, 1)]
        sample = dataset[0]
        self.assertEqual(sample[1].tolist(), [0.0, 2.0, 0.0, 0.0])
        self.assertEqual(sample[3].tolist(), [1.0, 0.0, 0.0, 0.0])

        buffer = Critic_Buffer_Reward.__new__(Critic_Buffer_Reward)
        buffer.gamma = 0.99
        buffer.lam = 0.95

        class ZeroValue(torch.nn.Module):
            def forward(self, observations):
                return torch.zeros(observations.shape[:-1])

        _, target = buffer.obtain_training_data(
            ZeroValue(), tuple(value.unsqueeze(0) for value in sample), "cpu"
        )
        self.assertGreater(target.item(), 1.8)

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
        tuner.adjoint_k = -torch.tensor([0.3, 0.5, 0.7, 0.9])
        tuner.new_score_net = Score(0.2)
        tuner.old_score_net = Score(-0.1)
        trajectory = [torch.randn(1, 3, 2) for _ in range(4)]
        adjoints = [torch.randn(1, 3, 2) for _ in range(4)]
        vectorized = tuner.adjoint_matching_loss(trajectory, adjoints)
        scalar = torch.tensor(0.0)
        for index, (state, adjoint) in enumerate(zip(trajectory, adjoints)):
            k_value = tuner.adjoint_k[index]
            timestep = tuner.t_asc[index : index + 1]
            new_v = k_value * state + k_value * tuner.new_score_net(state, timestep)
            old_v = k_value * state + k_value * tuner.old_score_net(state, timestep)
            sigma = torch.sqrt(-2 * k_value)
            value = ((new_v - old_v) * (2 / sigma) + sigma * adjoint).square().mean()
            if index <= 1:
                value = torch.minimum(value, torch.tensor(14.4))
            scalar += value
        self.assertTrue(torch.allclose(vectorized, scalar / 4, atol=1e-6))

    def test_constraint_penalty_uses_average(self):
        model = TotalReward_Critic.__new__(TotalReward_Critic)
        torch.nn.Module.__init__(model)
        model.config = types.SimpleNamespace(
            d_s=2, d_a=1, critic_d_s=2, critic_gamma=0.99,
            delta=torch.tensor(0.25), device=torch.device("cpu"),
        )
        model.reward_processor = model.kernel_processor = model.critic_processor = lambda x: x
        model.sigmoid = lambda *args: torch.ones(1)
        model.reward_net = lambda *args: torch.zeros(1)
        model.critic = lambda *args: torch.zeros(1)
        model.q_stats = types.SimpleNamespace(Q_mean=0.0, Q_std=5.0)
        plan = torch.zeros(4, 3)
        constraint = model.get_c(plan)
        objective = model.predict(plan, 2.0)
        self.assertTrue(torch.allclose(objective, -2.0 * constraint))


if __name__ == "__main__":
    unittest.main()
