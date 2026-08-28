import importlib.machinery
import sys
import types
import unittest

import numpy as np
import torch


if "wandb" not in sys.modules:
    wandb = types.ModuleType("wandb")
    wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    wandb.init = lambda *args, **kwargs: None
    wandb.log = lambda *args, **kwargs: None
    wandb.finish = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb

from Finetuning.acc_adjoint_matching import Acc_AdjointMatchingFineTuner
from Finetuning.traj_reward4 import TotalReward_Critic
from Finetuning.utils import CriticDataset_Reward, Critic_Buffer_Reward
from Pretrain.Critic.nets import Critic
from Pretrain.Planners.Backbone.Sampler import sample_euler_karras_batch
from Pretrain.Rewards.nets import SimpleReward


class _ZeroScore(torch.nn.Module):
    def forward(self, x, timestep):
        if timestep.shape != (x.shape[0],):
            raise AssertionError((x.shape, timestep.shape))
        return torch.zeros_like(x)


class PipelineRegressionTest(unittest.TestCase):
    def test_batched_sampler_conditions_every_candidate(self):
        initial_state = np.array([0.25, -0.5, 1.0], dtype=np.float32)
        plans = sample_euler_karras_batch(
            initial_state,
            _ZeroScore(),
            d_s=3,
            d_a=2,
            horizon=5,
            num_steps=3,
            num_karras=1,
            eta=0.0,
            device="cpu",
            num_samples=4,
        )
        self.assertEqual(plans.shape, (4, 5, 5))
        np.testing.assert_allclose(
            plans[:, 0, :3], np.repeat(initial_state[None, :], 4, axis=0)
        )
        self.assertLessEqual(float(np.abs(plans[..., 3:]).max()), 1.0 + 1e-6)

    def test_reward_and_critic_are_nonnegative(self):
        critic = Critic(3, hidden_dim=16, hidden_layers=1)
        reward = SimpleReward(3, 2, hidden_dim=16, hidden_layers=1)
        self.assertTrue(torch.all(critic(torch.randn(64, 3)) >= 0))
        self.assertTrue(torch.all(reward(torch.randn(64, 3), torch.randn(64, 2)) >= 0))

    def test_short_trajectory_keeps_terminal_reward(self):
        dataset = CriticDataset_Reward.__new__(CriticDataset_Reward)
        dataset.horizon = 4
        dataset._normalized_observations = [
            np.array([[0.0], [1.0]], dtype=np.float32)
        ]
        dataset._scaled_rewards = [np.array([0.0, 10.0], dtype=np.float32)]
        dataset._windows = [(0, 0), (0, 1)]

        sample = dataset[0]
        self.assertEqual(sample[0].shape, (5, 1))
        self.assertEqual(sample[1].tolist(), [0.0, 10.0, 0.0, 0.0])
        self.assertEqual(sample[2].tolist(), [1.0, 1.0, 0.0, 0.0])
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
        self.assertGreater(target.item(), 9.0)

    def test_vectorized_adjoint_loss_matches_scalar_formula(self):
        class Score(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(scale))

            def forward(self, x, timestep):
                return self.scale * x + timestep[:, None, None]

        tuner = Acc_AdjointMatchingFineTuner.__new__(Acc_AdjointMatchingFineTuner)
        tuner.device = torch.device("cpu")
        tuner.config = types.SimpleNamespace(
            num_Loss_Clip_steps=1, reward_scaling_factor=3.0
        )
        tuner.t_asc = torch.linspace(0.9, 0.1, 4)
        tuner.adjoint_k = -torch.tensor([0.3, 0.5, 0.7, 0.9])
        tuner.new_score_net = Score(0.2)
        tuner.old_score_net = Score(-0.1)
        for parameter in tuner.old_score_net.parameters():
            parameter.requires_grad_(False)

        trajectory = [torch.randn(1, 3, 2) for _ in range(4)]
        adjoints = [torch.randn(1, 3, 2) for _ in range(4)]
        vectorized = tuner.adjoint_matching_loss(trajectory, adjoints)

        scalar = torch.tensor(0.0)
        clip_value = torch.tensor(3.0**2 * 1.6)
        for index, (state, adjoint) in enumerate(zip(trajectory, adjoints)):
            k_value = tuner.adjoint_k[index]
            timestep = tuner.t_asc[index : index + 1]
            new_v = k_value * state + k_value * tuner.new_score_net(state, timestep)
            old_v = k_value * state + k_value * tuner.old_score_net(state, timestep)
            sigma = torch.sqrt(-2 * k_value)
            value = ((new_v - old_v) * (2 / sigma) + sigma * adjoint).square().mean()
            if index <= tuner.config.num_Loss_Clip_steps:
                value = torch.minimum(value, clip_value)
            scalar += value
        scalar /= len(trajectory)

        self.assertTrue(torch.allclose(vectorized, scalar, atol=1e-6))

    def test_critic_constraint_uses_same_average_as_lambda_update(self):
        model = TotalReward_Critic.__new__(TotalReward_Critic)
        torch.nn.Module.__init__(model)
        model.config = types.SimpleNamespace(
            d_s=2,
            d_a=1,
            critic_d_s=2,
            critic_gamma=0.99,
            delta=torch.tensor(0.25),
            device=torch.device("cpu"),
        )
        model.reward_processor = lambda state: state
        model.kernel_processor = lambda state: state
        model.critic_processor = lambda state: state
        model.sigmoid = lambda state, action, next_state: torch.ones(1)

        class ZeroReward(torch.nn.Module):
            def forward(self, state, action):
                return torch.zeros(state.shape[0])

        class ZeroCritic(torch.nn.Module):
            def forward(self, state):
                return torch.zeros(state.shape[0])

        model.reward_net = ZeroReward()
        model.critic = ZeroCritic()
        model.Q_scale = types.SimpleNamespace(Q_scale=1.0)

        plan = torch.zeros(4, 3)
        lam = 2.0
        constraint = model.get_c(plan)
        objective = model.predict(plan, lam)
        self.assertTrue(torch.allclose(objective, -lam * constraint))


if __name__ == "__main__":
    unittest.main()
