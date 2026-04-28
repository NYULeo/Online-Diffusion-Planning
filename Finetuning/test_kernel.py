import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
from Kernels.Kernel_Backbone import test_kernel
from utils import get_trajs
import pickle
import random
import numpy as np
import torch

def set_seed(seed=0):
    # Python random
    random.seed(seed)
    # NumPy random
    np.random.seed(seed)
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    # PyTorch deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)




if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    #trajs = get_trajs('pointmaze', 'medium', step = 0)
    test_kernel(dataset_name = 'pointmaze', 
                specific_dataset = 'medium', 
                trajs = None, 
                save_freq = 0, 
                num_steps = 0, 
                ensemble_size = 10)
    #test_Model(dataset_name = 'kitchen', save_freq = 2000, num_steps = 300000)
    