
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Transition_Kernel.Kernel_Backbone import test_kernel, train_kernel, train_mog_kernel, test_kernel_mog
from utils import set_seed
from Finetuning.utils import get_trajs
import pickle

if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    #train_kernel(dataset_name = 'kitchen', batch_size = 512, lr = 1e-4, num_steps =  200, save_freq = 200, ensemble_size = 20, hidden_layers = 2, hidden_dim = 256, λ_reg = 5e-3)
    #trajs = get_trajs('kitchen', 'partial', step = 0)
    
    #test_Model(dataset_name = 'pointmaze', specific_dataset = 'medium', trajs = trajs, save_freq = 2000, num_steps = 50000)
    #test_kernel(dataset_name = 'kitchen', trajs = trajs, save_freq = 200, num_steps = 200, hidden_layers = 2, hidden_dim = 256, ensemble_size = 20)
    #train_kernel(dataset_name = 'pointmaze', specific_dataset ='umaze', batch_size = 256, lr = 3e-4, num_steps = 25000, ensemble_size=3, λ_reg=1e-3)
    #train_kernel(dataset_name = 'pointmaze', specific_dataset ='medium', batch_size = 256, lr = 3e-4, num_steps = 50000, ensemble_size=3, λ_reg=1e-3)
    #train_kernel(dataset_name = 'pointmaze', specific_dataset ='large', batch_size = 256, lr = 3e-4, num_steps = 300000, ensemble_size=3, λ_reg=1e-3)
    dataset = 'cube'
    specific_dataset = 'double'
    train_mog_kernel(
         dataset_name = dataset,
         specific_dataset = specific_dataset,
         batch_size = 512,
         lr = 1e-4,
         num_steps = 50000,
         save_freq = 10000,
         ensemble_size = 10,
         num_modes = 8,
         num_hidden_layers = 3,
         hidden_dim = 512,
         λ_reg = 2e-3,
         noise_floor = 5e-4)
      
    test_kernel_mog(dataset_name = dataset,
                specific_dataset = specific_dataset,
                trajs = None,
                save_freq = 50000,
                num_steps = 50000,
                num_hidden_layers = 3,
                hidden_dim = 512,
                ensemble_size = 10, 
                num_modes = 8,
                quantile = 0.95,
                noise_floor = 5e-4)
