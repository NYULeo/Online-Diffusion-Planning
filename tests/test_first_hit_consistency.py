import unittest
from unittest.mock import patch

import numpy as np
import torch

from Finetuning.utils import discounted_first_hit_returns, first_hit_cost_terms
from Finetuning import utils as finetuning_utils
from Pretrain.Dataset import CubeDataset_Singletask
from Pretrain.Rewards.nets import SimpleReward
from Pretrain.utils import SAStats


class FirstHitConsistencyTest(unittest.TestCase):
    def test_soft_first_hit_terms(self):
        rewards = torch.tensor([-1.0, -0.5, 0.0, -1.0])
        effective, survival = first_hit_cost_terms(rewards)
        torch.testing.assert_close(
            effective,
            torch.tensor([-1.0, -0.5, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            survival,
            torch.tensor([1.0, 0.5, 0.0, 0.0]),
        )

    def test_discounted_return_stops_at_success(self):
        returns = discounted_first_hit_returns(
            np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
            gamma=0.9,
        )
        np.testing.assert_allclose(
            returns,
            [-1.9, -1.0, 0.0, -1.0],
            rtol=1e-6,
        )

    def test_cube_tail_is_not_a_second_trajectory(self):
        dataset = CubeDataset_Singletask.__new__(CubeDataset_Singletask)
        dataset.traj_length = None
        dataset.dataset = {
            "observations": np.arange(12, dtype=np.float32).reshape(6, 2),
            "actions": np.arange(6, dtype=np.float32).reshape(6, 1),
            "rewards": np.array([-1, 0, 0, -1, -1, -1], dtype=np.float32),
            "masks": np.array([1, 0, 0, 1, 1, 1], dtype=np.float32),
            "terminals": np.array([0, 0, 0, 1, 0, 1], dtype=np.float32),
        }
        dataset.eval_dataset = {
            key: value.copy() for key, value in dataset.dataset.items()
        }

        trajectories = dataset.get_trajectories(split="train")
        self.assertEqual(len(trajectories), 2)
        np.testing.assert_array_equal(
            trajectories[0]["rewards"],
            [-1, 0],
        )
        np.testing.assert_array_equal(
            trajectories[1]["rewards"],
            [-1, -1],
        )

    def test_critic_dataset_assigns_a_target_to_every_state(self):
        obs_dim, act_dim = 2, 1
        reward_model = SimpleReward(obs_dim, act_dim, hidden_dim=4, hidden_layers=1)
        for parameter in reward_model.parameters():
            torch.nn.init.zeros_(parameter)
        stats = SAStats()
        stats.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        stats.obs_std = np.ones(obs_dim, dtype=np.float32)
        trajectories = [{
            "observations": np.zeros((3, obs_dim), dtype=np.float32),
            "actions": np.zeros((3, act_dim), dtype=np.float32),
            "rewards": np.array([-1, -1, 0], dtype=np.float32),
            "masks": np.array([1, 1, 0], dtype=np.float32),
        }]

        with (
            patch.object(
                finetuning_utils,
                "get_env",
                return_value=(None, obs_dim, act_dim),
            ),
            patch.object(
                finetuning_utils,
                "get_reward_model",
                return_value=(reward_model.state_dict(), obs_dim, act_dim),
            ),
            patch.object(
                finetuning_utils,
                "get_reward_stats",
                return_value=stats,
            ),
            patch.object(
                finetuning_utils,
                "check_device",
                return_value=torch.device("cpu"),
            ),
        ):
            dataset = finetuning_utils.CriticDataset_Reward(
                "cube",
                "single-play",
                reward_hidden_layers=1,
                reward_hidden_dim=4,
                reward_checkpoint=0,
                trajs=trajectories,
                gamma=0.9,
                value_scale=1.0,
                stats=stats,
                save_stats=False,
            )

        self.assertEqual(len(dataset), 3)
        self.assertEqual(tuple(dataset[0][0].shape), (obs_dim,))


if __name__ == "__main__":
    unittest.main()
