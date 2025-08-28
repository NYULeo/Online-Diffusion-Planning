from typing import Optional
from Dataset import KitchenDataset,  PointMazeDataset



def get_env(dataset_name: str, specific_dataset:  Optional[str] = None):
     if(dataset_name == 'kitchen'):
        return KitchenDataset('complete').get_env()
     elif(dataset_name == 'pointmaze'):
         if(specific_dataset == 'large'):
              return PointMazeDataset('large').get_env()
         elif(specific_dataset == "medium"):
              return PointMazeDataset('medium').get_env()
         elif(specific_dataset == 'umaze'):
              return PointMazeDataset('umaze').get_env()
         else:
              raise ValueError(f"Invalid specific dataset: {dataset_name}")

     else: 
         raise ValueError(f"Invalid dataset name: {dataset_name}")