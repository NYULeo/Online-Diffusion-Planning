
from Transition_Kernel.Kernel_Backbone import train_kernel
from utils import set_seed


if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    train_kernel(dataset_name = 'kitchen', batch_size = 256, lr = 1e-4, num_steps =  50000, ensemble_size=3, λ_reg=1e-3)
    #train_kernel(dataset_name = 'pointmaze', specific_dataset ='umaze', batch_size = 256, lr = 3e-4, num_steps = 25000, ensemble_size=3, λ_reg=1e-3)
    #train_kernel(dataset_name = 'pointmaze', specific_dataset ='medium', batch_size = 256, lr = 3e-4, num_steps = 50000, ensemble_size=3, λ_reg=1e-3)
    #train_kernel(dataset_name = 'pointmaze', specific_dataset ='large', batch_size = 256, lr = 3e-4, num_steps = 300000, ensemble_size=3, λ_reg=1e-3)