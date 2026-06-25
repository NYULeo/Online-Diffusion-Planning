

from utils import set_seed
from Planners.Backbone.Trainer import SDETrainer




if __name__ == '__main__':  # pragma: no cover
     rng = set_seed(1)  # set_seed now returns a jax PRNGKey (CONVERSION_GUIDE §8); thread it onward.
     dataset_name = 'cube'
     specific_dataset = 'triple-play'
     task_id = 5
     horizon = 32
     trainer = SDETrainer(
         dataset_name,
         specific_dataset,
         task_id,
         horizon,
         backbone_name = 'transformer',
         num_steps = 1000000,
         batch_size = 128,
         lr = 2e-4,
         stride = 1)
     trainer.train()
     #trainer.selector('complete', times = 1000)
