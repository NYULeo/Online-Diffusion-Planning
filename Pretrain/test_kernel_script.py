import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Transition_Kernel.Kernel_Backbone import test_kernel
from utils import set_seed
from Finetuning.utils import get_trajs
import pickle





if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    
    test_kernel(dataset_name = 'pointmaze', specific_dataset = 'medium', trajs = trajs, save_freq = 2000, num_steps = 50000, ensemble_size = 10)
    #test_Model(dataset_name = 'kitchen', save_freq = 2000, num_steps = 300000)
    