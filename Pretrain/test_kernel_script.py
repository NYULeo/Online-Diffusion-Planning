from Transition_Kernel.Kernel_Backbone import test_Model
from utils import set_seed




if __name__ == '__main__':  # pragma: no cover
    set_seed(1)
    test_Model(dataset_name = 'kitchen', save_freq = 5000, num_steps = 300000)