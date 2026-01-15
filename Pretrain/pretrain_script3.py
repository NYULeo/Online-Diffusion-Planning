from utils import set_seed
from Planners.Backbone.Trainer import SDETrainer
import torch



if __name__ == '__main__':  # pragma: no cover
     set_seed(1)
     dataset_name = 'antmaze'
     specific_dataset = 'medium_diverse'
     horizon = 40
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     trainer = SDETrainer(
         dataset_name, 
         specific_dataset, 
         horizon, 
         backbone_name = 'transformer',
         num_steps = 1000000, 
         batch_size = 128,
         lr = 2e-4,
         device = device)
     trainer.train()
     #trainer.selector('complete', times = 1000)


