import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle


# Dataset helpers 
class TrajectoryDataset(Dataset):
    def __init__(self, npy_path: str):
        try:
            arr = np.load(npy_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Trajectory file not found: {npy_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading trajectory file {npy_path}: {e}")
        
        assert arr.ndim == 3, "Expected [N,H,D] numpy array"
        self.trajs = arr.astype(np.float32)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trajs[idx])


class HumanoidBenchDataset(Dataset):
    """Dataset class for HumanoidBench data with 14 tasks from local hbench.pickle file."""
    
    def __init__(self, data_path: str, horizon: int = 64, stride: int = 8, 
                 selected_tasks: list = None, task_conditioning: bool = True):
        """
        Initialize HumanoidBench dataset.
        
        Args:
            data_path: Path to HumanoidBench pickle file
            horizon: Length of trajectory windows
            stride: Stride between trajectory windows
            selected_tasks: List of task names to include (if None, include all 14 tasks)
            task_conditioning: Whether to include task conditioning information
        """
        self.data_path = data_path
        self.horizon = horizon
        self.stride = stride
        self.task_conditioning = task_conditioning
        
        # Define all 14 HumanoidBench tasks
        self.all_tasks = [
            'h1-run-v0', 'h1-walk-v0', 'h1-stand-v0', 'h1-reach-v0', 
            'h1-balance_hard-v0', 'h1-sit_simple-v0', 'h1-stair-v0', 'h1-sit_hard-v0',
            'h1-maze-v0', 'h1-crawl-v0', 'h1-balance_simple-v0', 'h1-hurdle-v0',
            'h1-pole-v0', 'h1-slide-v0'
        ]
        
        # Filter tasks if specified
        if selected_tasks is None:
            self.selected_tasks = self.all_tasks
        else:
            self.selected_tasks = [task for task in selected_tasks if task in self.all_tasks]
            if len(self.selected_tasks) != len(selected_tasks):
                print(f"⚠️  Warning: Some tasks not found. Using: {self.selected_tasks}")
        
        print(f"🎯 Selected tasks ({len(self.selected_tasks)}): {self.selected_tasks}")
        
        # Load and process data
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """Load pickle data and extract trajectories."""
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading pickle file {self.data_path}: {e}")
        
        print(f"📊 Dataset contains {len(data)} tasks")
        print(f"🔑 Available tasks: {list(data.keys())}")
        
        all_trajectories = []
        all_task_conditions = []
        
        for task_idx, task_name in enumerate(self.selected_tasks):
            if task_name not in data:
                print(f"⚠️  Warning: Task {task_name} not found in dataset")
                continue
            print(f"🔄 Processing task {task_idx+1}/{len(self.selected_tasks)}: {task_name}")
            task_data = data[task_name]
            
            for episode_idx in [0, 1, 2]:
                if episode_idx not in task_data:
                    continue
                episode_data = task_data[episode_idx]
                observations = episode_data['observation']
                actions = episode_data['action']
                joint = np.concatenate([observations, actions], axis=1)
                
                T = joint.shape[0]
                if T >= self.horizon:
                    for start in range(0, T - self.horizon + 1, self.stride):
                        trajectory = joint[start:start + self.horizon]
                        all_trajectories.append(trajectory)
                        if self.task_conditioning:
                            task_condition = np.zeros(len(self.selected_tasks))
                            task_condition[task_idx] = 1.0
                            all_task_conditions.append(task_condition)
        
        if not all_trajectories:
            raise ValueError("No valid trajectories extracted")
        
        self.trajectories = np.array(all_trajectories, dtype=np.float32)
        print(f"✅ Extracted {len(self.trajectories)} trajectories with shape {self.trajectories.shape}")
        
        if self.task_conditioning:
            self.task_conditions = np.array(all_task_conditions, dtype=np.float32)
            print(f"✅ Created task conditions: {self.task_conditions.shape}")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = torch.from_numpy(self.trajectories[idx])
        if self.task_conditioning and hasattr(self, 'task_conditions'):
            task_condition = torch.from_numpy(self.task_conditions[idx])
            return trajectory, task_condition
        else:
            return trajectory


def get_task_specific_dataset(task_name: str, data_path: str = "../hbench.pickle", horizon: int = 64, stride: int = 8):
    """
    Get dataset for a specific task without task conditioning.
    
    Args:
        task_name: Name of the task (e.g., 'h1-run-v0')
        data_path: Path to HumanoidBench pickle file
        horizon: Length of trajectory windows
        stride: Stride between trajectory windows
        
    Returns:
        Path to preprocessed numpy file for the specific task
    """
    # Check if the local file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HumanoidBench dataset not found at {data_path}. Please ensure hbench.pickle is in the parent directory.")
    
    # Create dataset for specific task without conditioning
    dataset = HumanoidBenchDataset(data_path, horizon=horizon, stride=stride, 
                                  selected_tasks=[task_name], task_conditioning=False)
    
    # Save task-specific data
    preprocessed_path = data_path.replace('.pickle', f'_{task_name}_trajectories.npy')
    np.save(preprocessed_path, dataset.trajectories)
    print(f"✅ Saved task-specific data: {preprocessed_path}")
    
    return preprocessed_path


def prepare_humanoidbench_for_training(data_path: str = "../hbench.pickle", horizon: int = 64, stride: int = 8, 
                                      selected_tasks: list = None, task_conditioning: bool = True):
    """Prepare HumanoidBench dataset for training from local hbench.pickle file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HumanoidBench dataset not found at {data_path}. Please ensure hbench.pickle is in the parent directory.")
    
    dataset = HumanoidBenchDataset(data_path, horizon=horizon, stride=stride, 
                                  selected_tasks=selected_tasks, task_conditioning=task_conditioning)
    
    # Save preprocessed data
    preprocessed_path = data_path.replace('.pickle', '_trajectories.npy')
    np.save(preprocessed_path, dataset.trajectories)
    print(f"✅ Saved preprocessed data: {preprocessed_path}")
    
    if task_conditioning and hasattr(dataset, 'task_conditions'):
        task_cond_path = data_path.replace('.pickle', '_task_conditions.npy')
        np.save(task_cond_path, dataset.task_conditions)
        print(f"✅ Saved task conditions: {task_cond_path}")
    
    return preprocessed_path



