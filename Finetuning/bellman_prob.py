

#!/usr/bin/env python3
"""Probe multi-horizon R^K consistency. Single GPU/CPU. Repo root cwd."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.chdir(project_root)

from Pretrain.Dataset import get_dataset
from Finetuning.Raw import probe_multi_horizon_bellman  # paste the probe into utils.py first


def main():
    dataset_name = "cube"
    specific_dataset = "single-play"
    task_id = 4
    horizon = 32

    planner_checkpoint = 0
    reward_checkpoint = 0
    critic_checkpoint = 0

    data = get_dataset(dataset_name, specific_dataset, task_id=task_id)
    trajs = data.get_trajectories()

    stats, R = probe_multi_horizon_bellman(
        trajs=trajs,
        dataset_name=dataset_name,
        specific_dataset=specific_dataset,
        planner_checkpoint=planner_checkpoint,
        reward_checkpoint=reward_checkpoint,
        critic_checkpoint=critic_checkpoint,
        backbone_layers=4,
        hidden_layers=4,
        hidden_dim=512,
        reward_hidden_layers=4,
        reward_hidden_dim=512,
        batch_size=256,
        oversample=20,
        horizon=horizon,
        gamma=0.99,
        steps_T=10,
        num_karras=1,
        eta=0.0,
        task_id=task_id,
        mix_reset=True,
        n_reset=64,
    )
    

    """
    out = os.path.join(
        project_root, "Finetuning", "logs",
        f"bellman_probe_{dataset_name}_{specific_dataset}_task{task_id}.pt",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import torch
    torch.save({"stats": stats, "R": R}, out)
    print("saved", out)
    """


if __name__ == "__main__":
    main()