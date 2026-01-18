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
    trajs = get_trajs('pointmaze', 'large', step = 0)
   
    
   