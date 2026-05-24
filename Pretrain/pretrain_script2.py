

from utils import set_seed
from Planners.Backbone.Trainer import SDETrainer
import torch
 


if __name__ == '__main__':  # pragma: no cover
     set_seed(1)
     dataset_name = 'ogpointmaze'
     specific_dataset = 'medium'
     horizon = 80
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     trainer = SDETrainer(
         dataset_name, 
         specific_dataset, 
         horizon,
         backbone_name = 'transformer',
         num_steps = 2000000, 
         batch_size = 128,
         lr = 3e-4,
         device = device,
         stride = 1)
     trainer.train()
     #trainer.selector('complete', times = 1000)


