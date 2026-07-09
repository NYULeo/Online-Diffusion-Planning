

from utils import set_seed
from Planners.Backbone.Trainer import SDETrainer
import torch





if __name__ == '__main__':  # pragma: no cover
     set_seed(1)
     dataset_name = 'cube'
     specific_dataset = 'triple-play'
     task_id = 5
     horizon = 32
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     trainer = SDETrainer(
         dataset_name, 
         specific_dataset, 
         task_id,
         horizon, 
         backbone_name = 'transformer',
         backbone_layers = 4,
         num_steps = 1000000, 
         batch_size = 128,
         lr = 2e-4,
         device = device,
         stride = 1)
     trainer.train()
     #trainer.selector('complete', times = 1000)


