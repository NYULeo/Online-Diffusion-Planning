

from utils import set_seed
from Backbone.Trainer import SDETrainer
import torch





if __name__ == '__main__':  # pragma: no cover
     set_seed(1)
     dataset_name = 'kitchen'
     specific_dataset = 'partial'
     horizon = 32
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     trainer = SDETrainer(dataset_name, specific_dataset, horizon, 
         num_steps = 100000, 
         batch_size = 32,
         lr=2e-4,
         device = device)
     trainer.train()


