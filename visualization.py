from operator import itemgetter

from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

from util import *


def visualize_groups(groups_dfs, dfs_groups, lw=50):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    fig_name = 'Classes and Groups'
    ax.set_title(fig_name, fontsize=15)
    all_alarms = pd.concat(map(lambda g_dfs: pd.concat(g_dfs[1]), groups_dfs))['HIT'].sort_index()
    n_samples = len(all_alarms)

    ax.scatter(range(n_samples), [3.5] * n_samples,
               c=all_alarms, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(n_samples), [.5] * n_samples,
               c=map(itemgetter(1), dfs_groups), marker='_', lw=lw, cmap=cmap_data)

    ax.legend([Patch(color=cmap_cv(.9)), Patch(color=cmap_data(0))],
              ['HIT', 'MISS'], loc=(1.02, .8))

    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['group', 'class'], xlabel="Sample index")
    plt.show()
    # plt.savefig(f'{fig_name}.png')


def plot_cv_indices(cv, splits, groups, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots()
    n_samples = sum(map(len, splits[0]))
    n_splits = len(splits)
    # Generate the training/testing visualizations for each CV split
    for i, (train_df, test_df) in enumerate(splits):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * n_samples)
        indices[train_df.index] = 0
        indices[test_df.index] = 1

        # Visualize the results
        ax.scatter(range(len(indices)), [i + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    all_alarms = pd.concat(splits[0])['HIT'].sort_index()
    ax.scatter(range(n_samples), [i + 1.5] * n_samples,
               c=all_alarms, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(n_samples), [i + 2.5] * n_samples,
               c=map(itemgetter(1), groups), marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 2.2, -.2], xlim=[0, n_samples])
    fig_name = '{}'.format(type(cv).__name__)
    ax.set_title(fig_name, fontsize=15)
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
    # Make the legend fit
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'{fig_name}.png')
