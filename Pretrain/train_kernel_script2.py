import os
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]  # Online-Diffusion-Planning/
sys.path.insert(0, str(REPO_ROOT))
#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#os.chdir(project_root)
from Pretrain.Transition_Kernel.Kernel_Backbone import train_kernel, test_kernel
from Pretrain.utils import set_seed
import pickle
from Finetuning.utils import get_trajs


if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    dataset = 'pointmaze'
    specific_dataset = 'large'
    path = REPO_ROOT / "Finetuning" / "Rollouts" / dataset / specific_dataset / "Generated_trajs_Info_0.pkl"
    with open(path, "rb") as f:
        trajs = pickle.load(f)
    #train_kernel(dataset_name = 'kitchen', batch_size = 256, lr = 1e-4, num_steps =  50000, ensemble_size=10, λ_reg=1e-3)
    #train_kernel(dataset_name = 'pointmaze', specific_dataset ='medium', batch_size = 256, lr = 3e-4, num_steps = 50000, ensemble_size=10, λ_reg=1e-3)
    train_kernel(dataset_name = dataset, specific_dataset = specific_dataset, batch_size = 256, lr = 3e-4, num_steps = 30000, save_freq = 3000, ensemble_size = 10, hidden_layers = 5, λ_reg = 1e-3)
    test_kernel(dataset_name = dataset, specific_dataset = specific_dataset,
                trajs = trajs, save_freq = 3000, num_steps = 30000, hidden_layers = 5, ensemble_size = 10)