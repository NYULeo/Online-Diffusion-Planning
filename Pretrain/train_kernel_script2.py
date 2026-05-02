import os
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]  # Online-Diffusion-Planning/
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#os.chdir(project_root)
from Pretrain.Transition_Kernel.Kernel_Backbone import train_kernel, test_kernel, train_mog_kernel, test_kernel_mog
from Pretrain.utils import set_seed
import pickle
from Finetuning.utils import get_trajs


if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    """
    dataset = 'pointmaze'
    specific_dataset = 'large'
    
    #train_kernel(dataset_name = 'kitchen', batch_size = 256, lr = 1e-4, num_steps =  50000, ensemble_size=10, λ_reg=1e-3)
    #train_kernel(dataset_name = 'pointmaze', specific_dataset ='medium', batch_size = 256, lr = 3e-4, num_steps = 50000, ensemble_size=10, λ_reg=1e-3)
    train_kernel(dataset_name = dataset, 
                 specific_dataset = specific_dataset, 
                 batch_size = 256, 
                 lr = 3e-4, 
                 num_steps = 1000, 
                 save_freq = 500, 
                 ensemble_size = 10, 
                 hidden_layers = 2, 
                 hidden_dim = 256,
                 λ_reg = 1e-3)
    """
    dataset = 'cube'
    specific_dataset = 'single'
    
    """
    train_kernel(dataset_name = dataset, 
                 specific_dataset = specific_dataset, 
                 batch_size = 256, 
                 lr = 3e-4, 
                 num_steps = 10000, 
                 save_freq = 5000, 
                 ensemble_size = 20, 
                 hidden_layers = 4, 
                 hidden_dim = 256,
                 λ_reg = 1e-3)
    """
    path = REPO_ROOT / "Finetuning" / "Rollouts" / dataset / 'single-play'/ f'task_{1}' / "Generated_trajs_Info_0.pkl"
    with open(path, "rb") as f:
          trajs = pickle.load(f)
    train_mog_kernel(
         dataset_name = dataset,
         specific_dataset = specific_dataset,
         trajs = trajs,
         batch_size = 512,
         lr = 1e-4,
         num_steps = 2000,
         save_freq = 500,
         ensemble_size = 10,
         num_modes = 5,
         num_hidden_layers = 3,
         hidden_dim = 514,
         λ_reg = 1e-3,
         noise_floor = 5e-4)
      
    test_kernel_mog(dataset_name = dataset,
                specific_dataset = specific_dataset,
                trajs = trajs,
                save_freq = 2000,
                num_steps = 2000,
                num_hidden_layers = 3,
                hidden_dim = 514,
                ensemble_size = 10, 
                num_modes = 5,
                quantile = 0.95,
                noise_floor = 5e-4)



    """
    path = REPO_ROOT / "Finetuning" / "Rollouts" / dataset / specific_dataset / "Generated_trajs_Info_0.pkl"
    with open(path, "rb") as f:
          trajs = pickle.load(f)
    test_kernel(dataset_name = dataset, 
                specific_dataset = specific_dataset,
                trajs = trajs, 
                save_freq = 500, 
                num_steps = 1000, 
                hidden_layers = 2, 
                hidden_dim = 256,
                ensemble_size = 10)
    """
    
    