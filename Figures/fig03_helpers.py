import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Compute histogram
import numpy as np
import matplotlib.pyplot as plt
import math

import numpy as np
import matplotlib.pyplot as plt
import math

import numpy as np
import matplotlib.pyplot as plt
import math

def plot_colored_histograms(
    sample_list,
    bins=100,
    figsize=None,
    nrows=None,
    ncols=None,
    cmap_name='viridis',
    title=None,
    subtitles=None,
    title_fontsize=16,
    subtitle_fontsize=12,
    axis_fontsize=10,
    display=False
):
    """
    Plot a colored histogram for each array in sample_list as subplots, with configurable titles and font sizes.

    Parameters
    ----------
    sample_list : list of 1D array-like
        Each element is a sample array to histogram.
    bins : int
        Number of bins for each histogram.
    figsize : tuple, optional
        Overall figure size (width, height). If None, computed automatically.
    nrows, ncols : int, optional
        Number of rows and columns of subplots. If both None, a near-square grid is chosen.
    cmap_name : str
        Name of the matplotlib colormap to use for bar coloring.
    title : str, optional
        Main title for the entire figure.
    subtitles : list of str, optional
        List of subtitles for each subplot; must match sample_list length.
    title_fontsize : int
        Font size for the main title.
    subtitle_fontsize : int
        Font size for each subplot's subtitle.
    axis_fontsize : int
        Font size for axis labels and subplot titles.
    """
    n = len(sample_list)
    if n == 0:
        raise ValueError("sample_list must contain at least one set of samples.")

    # Validate subtitles
    if subtitles is not None:
        if not isinstance(subtitles, (list, tuple)) or len(subtitles) != n:
            raise ValueError("subtitles must be a list of the same length as sample_list.")
    else:
        subtitles = [""] * n

    # Determine grid if not given
    if nrows is None and ncols is None:
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
    elif nrows is None:
        nrows = math.ceil(n / ncols)
    elif ncols is None:
        ncols = math.ceil(n / nrows)

    # Figure size heuristic: give each subplot a square-ish area
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    cmap = plt.get_cmap(cmap_name)

    # Set main title
    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=1.02)

    for idx, samples in enumerate(sample_list):
        i = idx // ncols
        j = idx % ncols
        ax = axes[i][j]

        # Compute histogram
        hist, bin_edges = np.histogram(samples, bins=bins, range=(-1, 1), density=True)
        hist = hist / hist.sum()

        # Map densities to colors
        normed = (hist - hist.min()) / (hist.max() - hist.min() + 1e-12)
        colors = cmap(normed)

        # Plot bars
        width = bin_edges[1] - bin_edges[0]
        ax.bar(
            bin_edges[:-1], hist,
            width=width,
            color=colors,
            align='edge',
            edgecolor='none',
            alpha=0.8
        )

        # Subplot title and subtitle
        ax.set_title(subtitles[idx], fontsize=subtitle_fontsize)


        ax.set_xlim(-1, 1)
        ax.set_xlabel('Action space', fontsize=axis_fontsize)
        ax.set_ylabel('Density', fontsize=axis_fontsize)
        ax.tick_params(axis='both', labelsize=int(axis_fontsize * 0.9))
        ax.grid(True, alpha=0.3)

    # Turn off any unused axes
    for j in range(n, nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])

    plt.tight_layout()
    if display:
        plt.show()
    return fig

def generate_mixed_gaussian_alternative(n_samples, means, stds, weights):
    """
    Alternative method using component selection with numpy.random.choice
    """
    # Choose which component each sample comes from
    component_indices = np.random.choice(len(means), size=n_samples, p=weights)
    
    # Generate samples from the chosen components
    samples = np.array([
        np.random.normal(means[i], stds[i]) 
        for i in component_indices
    ])
    
    return samples

