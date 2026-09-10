import numpy as np
import matplotlib.pyplot as plt
import ogbench





def plot_lines(data_list, labels=None, colors=None, markers=None, 
               xlabel='X', ylabel='Y', title='Plot', 
               save_path=None, xlim=None, ylim=None):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Default styling matching the image
    if colors is None:
        colors = ['#56B4E9',   # Light blue
                 '#CC79A7',  # Pink/magenta
                 '#0072B2']   # Darker blue
    
    if markers is None:
        markers = ['o', 's', 'D']  # Circle, square, diamond
    
    if labels is None:
        labels = [f'Series {i+1}' for i in range(len(data_list))]
    
    for idx, data in enumerate(data_list):
        # Convert list of [x, y] pairs to numpy arrays
        data = np.array(data)
        x = data[:, 0]
        y = data[:, 1]
        
        # Plot line with markers
        ax.plot(x, y, 
               color=colors[idx % len(colors)], 
               marker=markers[idx % len(markers)],
               markersize=6,
               linewidth=2,
               label=labels[idx],
               markeredgecolor='white' if markers[idx % len(markers)] == 'o' else colors[idx % len(colors)],
               markeredgewidth=0.5 if markers[idx % len(markers)] == 'o' else 1,
               markerfacecolor=colors[idx % len(colors)],
               fillstyle='full' if markers[idx % len(markers)] == 'o' else 'none')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


# Or manually create from the extracted values (x-axis divided by 100):
# Each round has 2400 env steps, divided by 100 = 24
env, dataset, eval_dataset = ogbench.make_env_and_datasets(
                "antmaze-giant-navigate-singletask-task5-v0", render_mode="rgb_array"
            )
print(max(dataset['rewards']))

