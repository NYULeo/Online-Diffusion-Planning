import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Transition_Kernel.Kernel_Backbone import test_kernel, test_kernel_mog
from utils import set_seed
from Finetuning.utils import get_trajs
import pickle





if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    #trajs = get_trajs('pointmaze', 'medium', step = 0)
    """
    test_kernel(dataset_name = 'pointmaze', 
                specific_dataset = 'medium', 
                trajs = None, 
                save_freq = 50000, 
                num_steps = 50000, 
                hidden_layers = 2, 
                hidden_dim = 256, 
                ensemble_size = 10)
    """
    """
    test_kernel(dataset_name = 'cube',
                specific_dataset = 'single',
                trajs = None,
                save_freq = 10000,
                num_steps = 10000,
                hidden_layers = 4,
                hidden_dim = 256,
                ensemble_size = 10)
    """
    test_kernel_mog(dataset_name = 'cube',
                specific_dataset = 'single',
                trajs = None,
                save_freq = 10000,
                num_steps = 10000,
                num_hidden_layers = 4,
                hidden_dim = 512,
                ensemble_size = 20, 
                num_modes = 8,
                quantile = 0.95)

    #test_Model(dataset_name = 'kitchen', save_freq = 2000, num_steps = 300000)
    