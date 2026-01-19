import numpy as np
import matplotlib.pyplot as plt





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

data = [
    [5561, 8.55],
    [8877, 14.13],
    [12110, 14.33],
    [14599, 30.54],
    [17081, 30.62],
    [19564, 30.62],
    [22016, 31.16],
    [24499, 30.71],
    [26965, 30.92],
    [29433, 30.81],
    [32002, 30.33],
    [34444, 31.12],
    [36877, 31.29],
    [39344, 30.74],
    [42054, 23.86],
    [45744, 19.73],
    [48200, 30.81],
    [52314, 16.07],
    [54736, 31.43],
    [57189, 30.83],
    [59634, 31.06],
    [62102, 30.79],
    [64554, 31.08],
    [67007, 31.03],
    [69466, 31.05],
    [71923, 31.07],
    [74395, 30.92],
    [76868, 30.91],
    [79321, 31.11],
    [81786, 31.00],
    [84244, 31.07],
    [86683, 31.24],
    [89110, 31.37],
    [91563, 31.15],
    [94015, 31.16],
    [96487, 30.92],
    [98950, 31.02],
    [101439, 30.65],
    [103841, 31.42],
]
print(data[27])
data = [
    [5561, 8.55],
    [8877, 14.13],
    [12110, 14.33],
    [14599, 30.54],
    [17081, 30.62],
    [19564, 30.62],
    [22016, 31.16],
    [24499, 30.71],
    [26965, 30.92],
    [29433, 30.81],
    [32002, 30.33],
    [34444, 31.12],
    [36877, 31.29],
    [39344, 30.74],
    [48200, 30.81],
    [54736, 31.43],
    [57189, 30.83],
    [59634, 31.06],
    [62102, 30.79],
    [64554, 31.08],
    [67007, 31.03],
    [69466, 31.05],
    [71923, 31.07],
    [74395, 30.92],
    [76868, 30.91],
    [79321, 31.11],
    [81786, 31.00],
    [84244, 31.07],
    [86683, 31.24],
    [89110, 31.37],
    [91563, 31.15],
    [94015, 31.16],
    [96487, 30.92],
    [98950, 31.02],
    [101439, 30.65],
    [103841, 31.42],
]

data = [
    [55, 8.55],
    [88, 14.13],
    [121, 14.33],
    [145, 30.54],
    [170, 30.62],
    [195, 30.62],
    [220, 31.16],
    [244, 30.71],
    [269, 30.92],
    [294, 30.81],
    [320, 30.33],
    [344, 31.12],
    [368, 31.29],
    [393, 30.74],
    [482, 30.81],
    [547, 31.43],
    [571, 30.83],
    [596, 31.06],
    [621, 30.79],
    [645, 31.08],
    [670, 31.03],
    [694, 31.05],
    [719, 31.07],
    [743, 30.92],
    [768, 30.91],
    [793, 31.11],
    [817, 31.00],
    [842, 31.07],
    [866, 31.24],
    [891, 31.37],
    [915, 31.15],
    [940, 31.16],
    [964, 30.92],
    [989, 31.02],
    [1014, 30.65],
    [1038, 31.42],
]
"""
plot_lines(
    [data],
    labels=['Normalized Score'],
    title='Normalized Score vs Environment Steps',
    xlabel='Env Steps (x100)',
    ylabel='Normalized Score'
)
"""
