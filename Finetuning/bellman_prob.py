import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.chdir(project_root)
from Pretrain.Dataset import get_dataset
from Finetuning.Raw import probe_multi_horizon_bellman  # paste the probe into utils.py first
from accelerate import Accelerator

def main():
    accelerator = Accelerator()
    os.chdir(project_root)
    dataset_name = "cube"
    specific_dataset = "single-play"
    task_id = 4
    horizon = 32
    reward_checkpoint = 0
    data = get_dataset(dataset_name, specific_dataset, task_id=task_id)
    trajs = data.get_trajectories()
          
    checkpoint = 0
    while checkpoint <= 0:
          planner_checkpoint = checkpoint
          critic_checkpoint = checkpoint
          print("-----------------------------------------------")
          print(f"checkpoint: {checkpoint}")
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
                  accelerator=accelerator,
             )
          print(stats)
          print(R)
          checkpoint += 3


if __name__ == "__main__":
    main()