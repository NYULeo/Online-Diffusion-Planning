import unittest
import pickle

import numpy as np
import torch
from torch.autograd.functional import jvp
import types
from unittest.mock import patch

from Finetuning.acc_adjoint_matching import Acc_AdjointMatchingFineTuner
from Finetuning.traj_reward4 import TotalReward, TotalReward_Critic, _normalization_tensors
from Finetuning.utils import (
    Critic_Test_Dataset,
    _compact_tensor_rows_for_object_gather,
    symexp,
    symlog,
)
from Pretrain.utils import SAStats, regression_diagnostics


class SafeOptimizationTest(unittest.TestCase):
    @staticmethod
    def freeze_module(module):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def test_symlog_symexp_round_trip(self):
        values = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        self.assertTrue(torch.allclose(symexp(symlog(values)), values, atol=1e-5))

    def test_regression_diagnostics_have_expected_direction(self):
        target = torch.tensor([0.0, 1.0, 2.0, 3.0])
        perfect = regression_diagnostics(target, target)
        collapsed = regression_diagnostics(torch.zeros_like(target), target)
        self.assertEqual(perfect["mae"], 0.0)
        self.assertEqual(perfect["normalized_mae"], 0.0)
        self.assertAlmostEqual(perfect["correlation"], 1.0, places=6)
        self.assertGreater(collapsed["normalized_mae"], perfect["normalized_mae"])
        self.assertLess(collapsed["std_ratio"], perfect["std_ratio"])

    def test_object_gather_rows_do_not_retain_full_backing_storage(self):
        plans = torch.randn(64, 8, 5)
        compact = _compact_tensor_rows_for_object_gather(plans)
        serialized_size = len(pickle.dumps(compact, protocol=pickle.HIGHEST_PROTOCOL))
        raw_size = plans.numel() * plans.element_size()
        self.assertLess(serialized_size, raw_size * 3)
        self.assertTrue(torch.equal(torch.stack(compact), plans))

    def test_critic_test_dataset_includes_exact_horizon_window(self):
        stats = SAStats()
        stats.obs_mean = np.zeros(2, dtype=np.float32)
        stats.obs_std = np.ones(2, dtype=np.float32)
        trajectory = {
            "observations": np.arange(8, dtype=np.float32).reshape(4, 2),
            "actions": np.zeros((4, 1), dtype=np.float32),
            "rewards": np.ones(4, dtype=np.float32),
        }
        with patch("Finetuning.utils.get_critic_stats", return_value=stats):
            dataset = Critic_Test_Dataset(
                "cube",
                "single-play",
                0,
                [trajectory],
                horizon=4,
            )
        self.assertEqual(len(dataset), 1)
        _, rewards = dataset[0]
        self.assertEqual(tuple(rewards.shape), (4,))

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
                continue
            scalar += value
        self.assertTrue(torch.allclose(vectorized, scalar / 4, atol=1e-6))

    def test_batched_adjoint_matches_scalar_trajectory_loop(self):
        class Score(torch.nn.Module):
            def forward(self, value, timestep):
                return 0.2 * value + timestep[:, None, None]

        class Reward:
            def __call__(self, plan, lam):
                return plan.square().sum() + lam, 2 * plan

        tuner = Acc_AdjointMatchingFineTuner.__new__(Acc_AdjointMatchingFineTuner)
        tuner.device = torch.device("cpu")
        tuner.old_score_net = Score()
        tuner.new_score_net = Score()
        tuner.config = types.SimpleNamespace(
            MaxEnt=False,
            reward_scaling_factor=5.0,
            Entropy_Scaling_Factor=0.5,
            num_Loss_Clip_steps=1,
        )
        tuner.Lam = types.SimpleNamespace(get_lam=lambda: 0.25)
        tuner.alpha_scheduler = types.SimpleNamespace(get_alpha=lambda: 0.8)
        tuner.t_asc = torch.linspace(0.9, 0.1, 4)
        tuner.k = -torch.tensor([0.3, 0.5, 0.7, 0.9])
        tuner.t_asc_reversed = torch.flip(tuner.t_asc, dims=[0])
        tuner.k_reversed = torch.flip(tuner.k, dims=[0])
        trajectories = torch.randn(3, 4, 1, 3, 2)

        scalar_adjoints = []
        scalar_rewards = []
        scalar_losses = []
        for trajectory_tensor in trajectories.clone():
            trajectory = [trajectory_tensor[index] for index in range(4)]
            adjoint, reward = tuner.make_a(trajectory, Reward(), reward_std=1.7)
            scalar_adjoints.append(torch.cat(adjoint, dim=0))
            scalar_rewards.append(reward)
            scalar_losses.append(tuner.adjoint_matching_loss(trajectory, adjoint))
        scalar_adjoints = torch.stack(scalar_adjoints)
        scalar_rewards = torch.stack(scalar_rewards)
        scalar_loss = torch.stack(scalar_losses).mean()

        batch_adjoints, batch_rewards = tuner.make_a_batch(
            trajectories.clone(), Reward(), reward_std=1.7
        )
        batch_loss = tuner.adjoint_matching_loss_batch(
            trajectories.clone(), batch_adjoints
        )
        self.assertTrue(torch.allclose(batch_adjoints, scalar_adjoints, atol=1e-6))
        self.assertTrue(torch.equal(batch_rewards, scalar_rewards))
        self.assertTrue(torch.allclose(batch_loss, scalar_loss, atol=1e-6))

    def test_vectorized_total_reward_matches_scalar_reference(self):
        class RewardNet(torch.nn.Module):
            def forward(self, state, action):
                return state.square().sum(dim=-1) + 0.3 * action.square().sum(dim=-1)

        class Critic(torch.nn.Module):
            def forward(self, state):
                return 0.4 * state.square().sum(dim=-1)

        model = TotalReward_Critic.__new__(TotalReward_Critic)
        torch.nn.Module.__init__(model)
        model.config = types.SimpleNamespace(
            d_s=2, d_a=1, critic_d_s=2, critic_gamma=0.9,
            delta=torch.tensor(0.2), device=torch.device("cpu"),
        )
        model.reward_obs_mean = torch.tensor([0.2, -0.3])
        model.reward_obs_std = torch.tensor([1.5, 0.7])
        model.kernel_obs_mean = torch.tensor([-0.1, 0.4])
        model.kernel_obs_std = torch.tensor([0.8, 1.2])
        model.critic_obs_mean = torch.tensor([0.5, -0.2])
        model.critic_obs_std = torch.tensor([1.1, 0.9])
        model.reward_net = RewardNet()
        model.critic = Critic()
        model.q_scale = types.SimpleNamespace(Q_scale=1.7)
        model.sigmoid = lambda state, action, next_state: (
            (next_state - state).square().sum(dim=-1)
            + 0.2 * action.square().sum(dim=-1)
        )
        plan = torch.tensor(
            [[0.1, -0.2, 1.4], [0.3, 0.5, -0.4], [-0.7, 0.8, 0.2], [0.9, -0.1, -1.3]],
            dtype=torch.float32,
        )
        lam = 0.35

        scalar_constraint = torch.tensor(0.0)
        scalar_total = torch.tensor(0.0)
        scalar_gradient = torch.zeros_like(plan)
        horizon = plan.shape[0]
        for index in range(horizon - 1):
            reward_state = model.reward_processor(plan[index, :2]).unsqueeze(0).requires_grad_(True)
            action = torch.clamp(plan[index, 2:].unsqueeze(0).requires_grad_(True), -1.0, 1.0)
            kernel_state = model.kernel_processor(plan[index, :2]).unsqueeze(0).requires_grad_(True)
            kernel_next = model.kernel_processor(plan[index + 1, :2]).unsqueeze(0).requires_grad_(True)
            reward = model.reward_net(reward_state, action)
            constraint = model.sigmoid(kernel_state, action, kernel_next)
            scalar_constraint += model.sigmoid(
                model.kernel_processor(plan[index, :2]).unsqueeze(0),
                plan[index, 2:].unsqueeze(0),
                model.kernel_processor(plan[index + 1, :2]).unsqueeze(0),
            ).squeeze(0)
            reward_state_grad, reward_action_grad = torch.autograd.grad(
                reward, (reward_state, action), torch.ones_like(reward)
            )
            constraint_state_grad, constraint_action_grad, constraint_next_grad = torch.autograd.grad(
                constraint, (kernel_state, action, kernel_next), torch.ones_like(constraint)
            )
            discount = model.config.critic_gamma**index
            scalar_gradient[index, :2] += discount * reward_state_grad.squeeze(0) / model.reward_obs_std
            scalar_gradient[index, 2:] += discount * reward_action_grad.squeeze(0)
            scalar_gradient[index, :2] -= lam * constraint_state_grad.squeeze(0) / model.kernel_obs_std
            scalar_gradient[index, 2:] -= lam * constraint_action_grad.squeeze(0)
            scalar_gradient[index + 1, :2] -= lam * constraint_next_grad.squeeze(0) / model.kernel_obs_std
            scalar_total += discount * reward.squeeze(0) - lam * constraint.squeeze(0)

        final_state = model.critic_processor(plan[-1, :2]).unsqueeze(0).requires_grad_(True)
        value = symexp(model.critic(final_state))
        value_grad = torch.autograd.grad(value, final_state, torch.ones_like(value))[0].squeeze(0)
        final_discount = model.config.critic_gamma ** (horizon - 1)
        scalar_gradient[-1, :2] += (
            final_discount * model.q_scale.Q_scale * value_grad / model.critic_obs_std
        )
        scalar_total += (
            final_discount * model.q_scale.Q_scale * value.squeeze(0)
            + lam * model.config.delta
        )
        scalar_constraint = scalar_constraint / (horizon - 1) - model.config.delta

        vector_constraint = model.get_c(plan)
        vector_prediction = model.predict(plan, lam)
        vector_total, vector_gradient = model(plan, lam)
        components = model.diagnostic_components_batch(plan, lam)
        self.assertTrue(torch.allclose(vector_constraint, scalar_constraint, atol=1e-6))
        self.assertTrue(torch.allclose(vector_prediction, scalar_total, atol=1e-6))
        self.assertTrue(torch.allclose(vector_total, scalar_total, atol=1e-6))
        self.assertTrue(torch.allclose(vector_gradient, scalar_gradient, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                components["compositional_reward"].squeeze(0),
                vector_prediction,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                components["base_reward"],
                components["immediate_reward"] + components["terminal_value"],
                atol=1e-6,
            )
        )

    def test_reward_only_diagnostics_match_predict(self):
        class RewardNet(torch.nn.Module):
            def forward(self, state, action):
                return state.sum(dim=-1) + 0.2 * action.sum(dim=-1)

        model = TotalReward.__new__(TotalReward)
        torch.nn.Module.__init__(model)
        model.config = types.SimpleNamespace(
            d_s=2,
            d_a=1,
            critic_gamma=0.9,
            delta=torch.tensor(0.2),
            device=torch.device("cpu"),
        )
        model.reward_obs_mean = torch.zeros(2)
        model.reward_obs_std = torch.ones(2)
        model.kernel_obs_mean = torch.zeros(2)
        model.kernel_obs_std = torch.ones(2)
        model.reward_net = RewardNet()
        model.sigmoid = lambda state, action, next_state: (
            (next_state - state).square().sum(dim=-1)
            + 0.1 * action.square().sum(dim=-1)
        )
        plan = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.4, -0.1], [0.5, 0.1, 0.2]],
            dtype=torch.float32,
        )
        prediction = model.predict(plan, 0.4)
        diagnostics = model.diagnostic_components_batch(plan, 0.4)
        self.assertTrue(
            torch.allclose(
                diagnostics["compositional_reward"].squeeze(0),
                prediction,
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
