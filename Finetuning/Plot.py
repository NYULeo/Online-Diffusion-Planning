import numpy as np
import matplotlib.pyplot as plt




"""
# Or manually create from the extracted values:
data = [
    [0, 9.22], [2400, 11.75], [4800, 11.66], [7200, 10.01], [9600, 11.39], 
    [12000, 13.75], [14400, 13.78], [16800, 13.61], [19200, 13.13], [21600, 12.49], 
    [24000, 13.94], [26400, 15.28], [28800, 14.38], [31200, 19.51], [33600, 18.37], 
    [36000, 17.73], [38400, 31.17], [40800, 31.43], [43200, 31.19], [45600, 31.36],
    [48000, 31.54], [50400, 31.51], [52800, 31.25], [55200, 31.50], [57600, 31.27],
    [60000, 31.44], [62400, 31.09], [64800, 31.07], [67200, 31.15], [69600, 31.09],
    [72000, 31.27]
]
"""

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
    [0, 9.22], [24, 11.75], [48, 11.66], [72, 10.01], [96, 11.39], 
    [120, 13.75], [144, 13.78], [168, 13.61], [192, 13.13], [216, 12.49], 
    [240, 13.94], [264, 15.28], [288, 14.38], [312, 19.51], [336, 18.37], 
    [360, 17.73], [384, 31.17], [408, 31.43], [432, 31.19], [456, 31.36],
    [480, 31.54], [504, 31.51], [528, 31.25], [552, 31.50], [576, 31.27],
    [600, 31.44], [624, 31.09], [648, 31.07], [672, 31.15], [696, 31.09],
    [720, 31.27]
]

data = [
    [0, 9.22],      # Initial rollout
    [24, 11.22],    # Round 1
    [48, 9.52],     # Round 2
    [72, 18.28],    # Round 3
    [96, 12.48],    # Round 4
    [120, 12.98],   # Round 5
    [144, 22.23],   # Round 6
    [168, 30.65],   # Round 7
    [192, 30.47],   # Round 8
    [216, 30.73],   # Round 9
    [240, 30.91],   # Round 10
    [264, 30.83],   # Round 11
    [288, 30.98],   # Round 12
    [312, 30.89],   # Round 13
    [336, 30.47],   # Round 14
    [360, 31.10],   # Round 15
    [384, 30.85],   # Round 16
    [408, 30.92],   # Round 17
    [432, 30.64],   # Round 18
    [456, 30.60],   # Round 19
    [480, 30.62]    # Round 20
]
plot_lines(
    [data],
    labels=['Normalized Score'],
    title='Normalized Score vs Environment Steps',
    xlabel='Env Steps (×100)',
    ylabel='Normalized Score'
)
