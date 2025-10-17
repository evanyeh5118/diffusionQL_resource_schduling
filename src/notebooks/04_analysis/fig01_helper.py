import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def visualize_reward_curves_grouped(curves_list, labels=None,
                                    ax=None, title=None, figsize=(8,5), show_legend=True,
                                    title_fontsize=12, label_fontsize=10, tick_fontsize=9, 
                                    legend_fontsize=9):
    """
    Plots each group on the provided ax. If ax is None, makes a new figure+ax.
    """
    if len(labels) != len(curves_list):
        raise ValueError("`labels` must have same length as `curves_list`")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    final_means = []
    final_mean_colors = []

    for g, (curves, lbl) in enumerate(zip(curves_list, labels)):
        series = []
        for key in sorted(curves, key=lambda k: int(k.split("_")[-1])):
            if not key.startswith("exp_"):
                continue
            arr = np.asarray(curves[key])
            if arr.ndim != 1:
                raise ValueError(f"{key!r} has invalid shape {arr.shape}; expected 1-D")
            series.append(arr)

        if not series:
            continue

        # align lengths
        min_len = min(len(a) for a in series)
        series = [a[:min_len] for a in series]

        mean_ts = np.mean(series, axis=0)
        lower_ts = np.min(series, axis=0)
        upper_ts = np.max(series, axis=0)

        x = np.arange(len(mean_ts))
        color = colors[g % len(colors)]
        ax.plot(x, mean_ts, label=lbl, color=color, linewidth=5)
        ax.fill_between(x, lower_ts, upper_ts, alpha=0.2, color=color)

        final_mean = mean_ts[-1]
        final_means.append(final_mean)
        final_mean_colors.append(color)

    ax.set_xlabel("Episode", fontsize=label_fontsize)
    ax.set_ylabel("PLR", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if show_legend:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), fontsize=legend_fontsize, ncol=len(labels))
    ax.grid(True)

    return ax

def plot_two_groups(curves_list1, labels1,
                    curves_list2, labels2,
                    figsize=(16,5), title1="Group 1", title2="Group 2", 
                    title_fontsize=12, label_fontsize=10, tick_fontsize=9, 
                    legend_fontsize=9):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot without individual legends
    visualize_reward_curves_grouped(curves_list1, labels1,
                                    ax=ax1, title=title1, show_legend=False,
                                    title_fontsize=title_fontsize, label_fontsize=label_fontsize,
                                    tick_fontsize=tick_fontsize, legend_fontsize=legend_fontsize)
    visualize_reward_curves_grouped(curves_list2, labels2,
                                    ax=ax2, title=title2, show_legend=False,
                                    title_fontsize=title_fontsize, label_fontsize=label_fontsize,
                                    tick_fontsize=tick_fontsize, legend_fontsize=legend_fontsize)
    
    # Create a single legend at the bottom
    # Get all handles and labels from both axes
    handles1, labels1_legend = ax1.get_legend_handles_labels()
    
    # Combine handles and labels, avoiding duplicates
    all_handles = []
    all_labels = []
    seen_labels = set()
    
    for handle, label in zip(handles1, labels1_legend):
        if label not in seen_labels:
            all_handles.append(handle)
            all_labels.append(label)
            seen_labels.add(label)
    
    # Add legend at the bottom of the figure
    fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
               ncol=int(len(all_handles)), frameon=True, fontsize=legend_fontsize)
    
    # Adjust layout to make room for the legend at the bottom
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    return fig

def load_reward_curves(paths, idxs, models_dir=""):
    reward_curves = {p: {} for p in paths}
    
    for p in paths:
        for idx in idxs:
            file_path = os.path.join(models_dir, p, f"train_metrics_{idx}_end.pkl")
            try:
                with open(file_path, "rb") as f:
                    metrics_train = pickle.load(f)
                reward_curves[p][f"exp_{idx}"] = metrics_train.get('Reward', [])
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
                continue
    
    return reward_curves

