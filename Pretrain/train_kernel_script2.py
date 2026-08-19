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
from Finetuning.utils import get_trajs, check_device

from Finetuning.Rollout import Test_Kernel_on_Generated_Trajs, Kernel_Config, save_success_trajs_for_reward, load_success_trajs


"""
if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    dataset = 'antmaze'
    specific_dataset = 'large'
    task_id = 4
    train_mog_kernel(
         dataset_name = dataset,
         specific_dataset = specific_dataset,
         task_id = task_id,
         batch_size = 512,
         lr = 1e-4,
         num_steps = 5000,
         save_freq = 1000,
         ensemble_size = 10,
         num_modes = 10,
         num_hidden_layers = 4,
         hidden_dim = 514,
         λ_reg = 1e-3,
         noise_floor = 5e-4)
      
    test_kernel_mog(
                dataset_name = dataset,
                specific_dataset = specific_dataset,
                task_id = task_id,
                trajs = None,
                save_freq = 5000,
                num_steps = 5000,
                num_hidden_layers = 4,
                hidden_dim = 514,
                ensemble_size = 10, 
                num_modes = 10,
                quantile = 0.99,
                noise_floor = 5e-4)
"""


if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    dataset = 'humanoidmaze'
    specific_dataset = 'large'
    task_id = 2
    train_mog_kernel(
         dataset_name = dataset,
         specific_dataset = specific_dataset,
         task_id = task_id,
         batch_size = 1024,
         lr = 1e-4,
         num_steps = 20000,
         save_freq = 20000,
         ensemble_size = 10,
         num_modes = 10,
         num_hidden_layers = 4,
         hidden_dim = 514,
         λ_reg = 1e-3,
         noise_floor = 5e-4)
      
    test_kernel_mog(
                dataset_name = dataset,
                specific_dataset = specific_dataset,
                task_id = task_id,
                trajs = None,
                save_freq = 20000,
                num_steps = 20000,
                num_hidden_layers = 4,
                hidden_dim = 514,
                ensemble_size = 10, 
                num_modes = 10,
                quantile = 0.99,
                noise_floor = 5e-4)