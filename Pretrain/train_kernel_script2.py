import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#os.chdir(project_root)
from Transition_Kernel.Kernel_Backbone import train_kernel, test_kernel
from utils import set_seed
import pickle
from Finetuning.utils import get_trajs


if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    dataset = 'pointmaze'
    specific_dataset = 'large'
    #train_kernel(dataset_name = 'kitchen', batch_size = 256, lr = 1e-4, num_steps =  50000, ensemble_size=10, λ_reg=1e-3)
    #train_kernel(dataset_name = 'pointmaze', specific_dataset ='medium', batch_size = 256, lr = 3e-4, num_steps = 50000, ensemble_size=10, λ_reg=1e-3)
    train_kernel(dataset_name = dataset, specific_dataset = specific_dataset, batch_size = 256, lr = 3e-4, num_steps = 30000, save_freq = 3000, ensemble_size = 10, hidden_layers = 5, λ_reg = 1e-3)
    test_kernel(dataset_name = dataset, specific_dataset = specific_dataset,
                trajs = None, save_freq = 3000, num_steps = 30000, hidden_layers = 5, ensemble_size = 10)