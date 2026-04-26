import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import ogbench as og
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import minari

from collections import deque
import gymnasium as gym
import gymnasium_robotics  # registers the envs
import numpy as np

# Example: UMaze, Medium, or Large
env = gym.make('PointMaze_Large-v3', max_episode_steps = 8000, render_mode = 'rgb_array', continuing_task=False)

# The maze object is inside the env
maze = env.unwrapped.maze  # or env.maze depending on version

def shortest_path_length(maze, start_xy, goal_xy):
    start = tuple(maze.cell_xy_to_rowcol(start_xy))
    goal  = tuple(maze.cell_xy_to_rowcol(goal_xy))
    
    if start == goal:
        return 0
    
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    queue = deque([(start, 0)])
    visited = set([start])
    
    while queue:
        (x, y), dist = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) == goal:
                return dist + 1
                
            if (0 <= nx < maze.map_length and 0 <= ny < maze.map_width and
                maze.maze_map[nx][ny] != 1 and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return float('inf')  # unreachable
start = np.array([6, 6])
goal  = np.array([6, 1])

length = shortest_path_length(maze, start, goal)
print(f"Minimum grid steps: {length}")



