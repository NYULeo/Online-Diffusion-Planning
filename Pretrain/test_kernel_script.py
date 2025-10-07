from Transition_Kernel.Kernel_Backbone import test_Model
from utils import set_seed
import pickle




if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    """
    with open('Rollouts/kitchen/partial/Generated_trajs_Info.pkl', 'rb') as f:
         trajs_info = pickle.load(f)
    trajs = trajs_info['trajs']
    """
    #test_Model(dataset_name = 'kitchen', trajs = trajs, save_freq = 5000, num_steps = 300000)
    test_Model(dataset_name = 'kitchen', save_freq = 5000, num_steps = 300000)