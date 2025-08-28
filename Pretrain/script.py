from train_planner import train_planner
from train_critic import train_critic
from train_kernel import train_kernel
from train_reward import train_reward
import random



if __name__ == '__main__':  # pragma: no cover
    random.seed(1)
    
    #Kitchen Environment
    train_critic(dataset_name = 'kitchen',  specific_dataset = 'partial',  batch_size=1024,  epochs=50,   gamma=0.99, lr=1e-3, tau = 0.005)
    train_critic(dataset_name = 'kitchen',  specific_dataset = 'mixed',  batch_size=1024,  epochs=50,   gamma=0.99, lr=1e-3, tau = 0.005)
    
    train_planner(dataset_name = 'kitchen', specific_dataset = 'partial', batch_size = 6, horizon = 32, num_epochs = 10, lr = 3e-4)
    train_planner(dataset_name = 'kitchen', specific_dataset = 'mixed', batch_size = 6, horizon = 32, num_epochs = 10, lr = 3e-4)
    
    
    #2DMaze Environment
    train_reward(dataset_name = 'pointmaze',  batch_size=1024, epochs=50,  lr=1e-3)
    


    train_planner(dataset_name = 'pointmaze', specific_dataset = 'large', batch_size = 6, horizon = 32, num_epochs = 10, lr = 3e-4)
    train_planner(dataset_name = 'pointmaze', specific_dataset = 'medium', batch_size = 6, horizon = 32, num_epochs = 10, lr = 3e-4)
    train_planner(dataset_name = 'pointmaze', specific_dataset = 'umaze', batch_size = 6, horizon = 32, num_epochs = 10, lr = 3e-4)
    
    

