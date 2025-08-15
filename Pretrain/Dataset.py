import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from typing import Optional, List, Dict, Any


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
    """Dataset class for HumanoidBench data with task-specific training from local hbench.pickle file."""
    
    def __init__(self, data_path: str, selected_tasks: list = None, max_trajectory_length: int = -1):
        """
        Initialize HumanoidBench dataset.
        
        Args:
            data_path: Path to HumanoidBench pickle file
            selected_tasks: List of task names to include (if None, include all 14 tasks)
            max_trajectory_length: Maximum trajectory length to use (-1 for full trajectory)
        """
        self.data_path = data_path
        self.max_trajectory_length = max_trajectory_length
        
        # Define all 14 HumanoidBench tasks
        self.all_tasks = [
            'h1-run-v0', 'h1-walk-v0', 'h1-stand-v0', 'h1-reach-v0', 
            'h1-balance_hard-v0', 'h1-sit_simple-v0', 'h1-stair-v0', 'h1-sit_hard-v0',
            'h1-maze-v0', 'h1-crawl-v0', 'h1-balance_simple-v0', 'h1-hurdle-v0',
            'h1-pole-v0', 'h1-slide-v0'
        ]
        
        # Use selected tasks or all tasks
        self.selected_tasks = selected_tasks if selected_tasks is not None else self.all_tasks
        print(f"🎯 Selected tasks ({len(self.selected_tasks)}): {self.selected_tasks}")
        
        # Load and process data
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """Load pickle data and extract full trajectories with task-specific dimensions."""
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading pickle file {self.data_path}: {e}")
        
        print(f"📊 Dataset contains {len(data)} tasks")
        print(f"🔑 Available tasks: {list(data.keys())}")
        
        all_trajectories = []
        task_dimensions = {}  # Store dimensions for each task
        
        for task_idx, task_name in enumerate(self.selected_tasks):
            if task_name not in data:
                print(f"⚠️  Warning: Task {task_name} not found in dataset")
                continue
            print(f"🔄 Processing task {task_idx+1}/{len(self.selected_tasks)}: {task_name}")
            task_data = data[task_name]
            
            # Get dimensions from first episode
            first_episode = task_data[0]
            obs_dims = first_episode['observation'].shape[1]
            act_dims = first_episode['action'].shape[1]
            joint_dims = obs_dims + act_dims
            
            task_dimensions[task_name] = {
                'observation_dims': obs_dims,
                'action_dims': act_dims,
                'joint_dims': joint_dims
            }
            
            print(f"   📏 Task dimensions: obs={obs_dims}, act={act_dims}, joint={joint_dims}")
            
            # Process all episodes for this task
            episodes = list(task_data.keys())
            for episode_idx in episodes:
                if episode_idx not in task_data:
                    continue
                episode_data = task_data[episode_idx]
                observations = episode_data['observation']
                actions = episode_data['action']
                
                # Concatenate observation and action for each step
                # Result: s1+a1, s2+a2, s3+a3, ..., sN+aN
                joint = np.concatenate([observations, actions], axis=1)
                
                # Use the full trajectory as provided (no windowing)
                trajectory_length = joint.shape[0]  # Number of (state, action) pairs
                print(f"   📏 Episode {episode_idx}: {trajectory_length:,} steps")
                
                # Truncate trajectory if max_trajectory_length is specified
                if self.max_trajectory_length > 0 and trajectory_length > self.max_trajectory_length:
                    joint = joint[:self.max_trajectory_length]
                    print(f"   ✂️  Truncated to {self.max_trajectory_length:,} steps")
                
                # Add the trajectory (full or truncated)
                all_trajectories.append(joint)
        
        if not all_trajectories:
            raise ValueError("No valid trajectories extracted")
        
        self.trajectories = [traj.astype(np.float32) for traj in all_trajectories]  # Ensure float32 type
        self.task_dimensions = task_dimensions
        
        # Verify all trajectories have the same dimensions
        if all_trajectories:
            unique_dims = set(traj.shape[1] for traj in all_trajectories)  # Use shape[1] for feature dimension
            if len(unique_dims) > 1:
                raise ValueError(f"Trajectories have different dimensions: {unique_dims}. "
                               f"This suggests mixing tasks with different observation spaces.")
        
        # Get trajectory length (should be same for all)
        trajectory_lengths = [traj.shape[0] for traj in all_trajectories]
        self.trajectory_length = trajectory_lengths[0]  # All should be same
        
        print(f"✅ Extracted {len(self.trajectories)} full trajectories")
        print(f"📏 Each trajectory: {self.trajectory_length:,} steps × {unique_dims.pop()} features")
        print(f"📊 Task dimensions summary: {task_dimensions}")
    
    def get_task_dimensions(self, task_name: str) -> Dict[str, int]:
        """Get dimensions for a specific task."""
        if task_name not in self.task_dimensions:
            raise ValueError(f"Task {task_name} not found in dataset")
        return self.task_dimensions[task_name]
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = torch.from_numpy(self.trajectories[idx])
        return trajectory


def get_task_specific_dataset(task_name: str, data_path: str = "hbench.pickle", max_trajectory_length: int = -1):
    """
    Get dataset for a specific task with automatic dimension detection.
    
    Args:
        task_name: Name of the task (e.g., 'h1-run-v0')
        data_path: Path to HumanoidBench pickle file
        max_trajectory_length: Maximum trajectory length to use (-1 for full trajectory)
        
    Returns:
        Tuple of (preprocessed_data_path, task_dimensions)
    """
    # Check if the local file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HumanoidBench dataset not found at {data_path}. Please ensure hbench.pickle is in the current directory.")
    
    # Create dataset for specific task
    dataset = HumanoidBenchDataset(data_path, selected_tasks=[task_name], max_trajectory_length=max_trajectory_length)
    
    # Get task dimensions
    task_dims = dataset.get_task_dimensions(task_name)
    
    # Save task-specific data
    preprocessed_path = data_path.replace('.pickle', f'_{task_name}_trajectories.npy')
    np.save(preprocessed_path, dataset.trajectories)
    print(f"✅ Saved task-specific data: {preprocessed_path}")
    print(f"📊 Task dimensions: {task_dims}")
    print(f"📏 Trajectory length: {dataset.trajectory_length:,} steps")
    
    return preprocessed_path, task_dims


def prepare_humanoidbench_for_training(data_path: str = "hbench.pickle", 
                                     selected_tasks: list = None):
    """Prepare HumanoidBench dataset for training from local hbench.pickle file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HumanoidBench dataset not found at {data_path}. Please ensure hbench.pickle is in the current directory.")
    
    dataset = HumanoidBenchDataset(data_path, selected_tasks=selected_tasks)
    
    # Save preprocessed data
    preprocessed_path = data_path.replace('.pickle', '_trajectories.npy')
    np.save(preprocessed_path, dataset.trajectories)
    print(f"✅ Saved preprocessed data: {preprocessed_path}")
    
    return preprocessed_path, dataset.task_dimensions



