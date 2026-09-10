

from utils import set_seed
from Planners.Backbone.Trainer import SDETrainer
import torch
 
def check_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using M3 GPU (MPS backend)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Falling back to CPU (no GPU acceleration)")
    return device 


if __name__ == '__main__':  # pragma: no cover
     set_seed(1)
     dataset_name = 'antmaze'
     specific_dataset = 'large'
     task_id = 4
     horizon = 32
     #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     device = check_device()
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


