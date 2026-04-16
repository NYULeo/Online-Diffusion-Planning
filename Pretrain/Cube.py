

import numpy as np
from typing import Optional, List, Dict
import ogbench


class CubeDataset:
    def __init__(self, name: str, task_id: Optional[int] = None, success_tol: float = 0.04):
        
        self.name = name
        self.success_tol = success_tol

        name_to_id = {
            "single-play": "cube-single-play-v0",
            "single-noisy": "cube-single-noisy-v0",
            "double-play": "cube-double-play-v0",
            "double-noisy": "cube-double-noisy-v0",
            "triple-play": "cube-triple-play-v0",
            "triple-noisy": "cube-triple-noisy-v0",
            "quadruple-play": "cube-quadruple-play-v0",
            "quadruple-noisy": "cube-quadruple-noisy-v0",
        }

        if name not in name_to_id:
            raise ValueError(f"Invalid dataset name: {name}")

        self.dataset_id = name_to_id[name]

        self.env, self.dataset, self.eval_dataset = ogbench.make_env_and_datasets(
            self.dataset_id, render_mode="rgb_array"
        )

        if task_id is not None:
            goal_xyzs = self.env.unwrapped.task_infos[task_id - 1]["goal_xyzs"]
            self.goal = goal_xyzs.reshape(-1).astype(np.float32)
            self.goal_dim = len(self.goal)
            self.num_cubes = self.goal_dim // 3
        else:
            self.goal = None
            self.goal_dim = None
            self.num_cubes = None

    def extract_cube_pos_vec(self, obs_vec, goal_dim: int) -> np.ndarray:
        """Extract concatenated XYZ positions of all cubes from observation."""
        num_cubes = goal_dim // 3
        obs_vec = np.asarray(obs_vec, dtype=np.float32).reshape(-1)

        expected_dim = 19 + 9 * num_cubes
        if obs_vec.shape[-1] != expected_dim:
            raise ValueError(
                f"Observation dimension mismatch for {num_cubes} cubes. "
                f"Got {obs_vec.shape[-1]}, expected {expected_dim}."
            )

        pos_parts = []
        for k in range(num_cubes):
            start = 19 + 9 * k
            pos_parts.append(obs_vec[start : start + 3])
        return np.concatenate(pos_parts, axis=0)  # shape (3*num_cubes,)

    def reached_goal_cube(self, goal_vec: np.ndarray, pos_vec: np.ndarray) -> bool:
        """Check if all cubes are within tolerance (OGBench success rule)."""
        goal = np.asarray(goal_vec, dtype=np.float32).reshape(-1, 3)
        pos = np.asarray(pos_vec, dtype=np.float32).reshape(-1, 3)

        if goal.shape != pos.shape:
            raise ValueError(f"Shape mismatch: goal {goal.shape}, pos {pos.shape}")

        dist = np.linalg.norm(pos - goal, axis=1)
        return bool(np.all(dist <= self.success_tol))

    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """Process trajectories with early truncation on first goal success."""
        trajectories = []
        last_start = 0
        N = len(self.dataset["observations"])

        for i in range(N):
            # End of a natural episode (terminal or dataset end)
            if self.dataset["terminals"][i] == 1 or i == N - 1:
                obs_slice = self.dataset["observations"][last_start : i + 1]
                act_slice = self.dataset["actions"][last_start : i + 1]

                if len(act_slice) < 10:
                    last_start = i + 1
                    continue

                if self.goal is not None:
                    rews = np.zeros(len(act_slice), dtype=np.float32)
                    success_idx = None

                    # Scan every step to find the FIRST time goal is reached
                    for t in range(len(act_slice)):
                        # Use next_observations because it reflects the state AFTER the action at step t
                        final_pos = self.extract_cube_pos_vec(
                            self.dataset["next_observations"][last_start + t],
                            self.goal_dim
                        )
                        if self.reached_goal_cube(self.goal, final_pos):
                            rews[t] = 1.0
                            success_idx = t
                            break  # Stop at first success → early truncation

                    if success_idx is not None:
                        # Truncate at the success step
                        obs_slice = obs_slice[: success_idx + 1]
                        act_slice = act_slice[: success_idx + 1]
                        rews = rews[: success_idx + 1]

                    trajectory = {
                        "observations": obs_slice,
                        "actions": act_slice,
                        "rewards": rews,
                    }
                else:
                    # No goal selected → all-zero rewards
                    rews = np.zeros(len(act_slice), dtype=np.float32)
                    trajectory = {
                        "observations": obs_slice,
                        "actions": act_slice,
                        "rewards": rews,
                    }

                trajectories.append(trajectory)
                last_start = i + 1

        return trajectories

    def get_state_dim(self) -> int:
        return int(self.dataset["observations"].shape[-1])

    def get_action_dim(self) -> int:
        return int(self.dataset["actions"].shape[-1])

    def get_env(self, render_mode: str = "rgb_array"):
        env, _, _ = ogbench.make_env_and_datasets(self.dataset_id, render_mode=render_mode)
        return env