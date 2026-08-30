
from utils import init_wandb_run, set_seed
from Planners.Backbone.Trainer import SDETrainer
import torch





if __name__ == '__main__':  # pragma: no cover
     set_seed(1)
     dataset_name = 'cube'
     specific_dataset = 'single-play'
     task_id = 4
     horizon = 32
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     run = init_wandb_run(
         "cube-single-task4-planner",
         {
             "stage": "planner_pretrain", "dataset_name": dataset_name,
             "specific_dataset": specific_dataset, "task_id": task_id,
             "horizon": horizon, "backbone_layers": 2,
             "num_steps": 1000000, "batch_size": 128, "lr": 2e-4,
         },
     )
     trainer = SDETrainer(
         dataset_name, 
         specific_dataset, 
         task_id,
         horizon, 
         backbone_name = 'transformer',
         backbone_layers = 2,
         num_steps = 1000000, 
         batch_size = 128,
         lr = 2e-4,
         device = device,
         stride = 1)
     try:
          trainer.train()
     finally:
          run.finish()
     #trainer.selector('complete', times = 1000)

