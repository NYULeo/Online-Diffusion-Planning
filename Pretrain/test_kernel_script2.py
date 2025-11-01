from Transition_Kernel.Kernel_Backbone import test_kernel
from utils import set_seed
import pickle




if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    with open('Rollouts/kitchen/partial/Generated_trajs_Info.pkl', 'rb') as f:
         trajs_info = pickle.load(f)
    trajs = trajs_info['trajs']
   
    #test_Model(dataset_name = 'pointmaze', specific_dataset = 'medium', trajs = trajs, save_freq = 2000, num_steps = 50000)
    #test_kernel(dataset_name = 'kitchen', trajs = trajs, save_freq = 2000, num_steps = 50000, ensemble_size = 10)