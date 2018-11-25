from operator import itemgetter
import logging

from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import Normalize

import numpy as np
import matplotlib.pyplot as plt

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

from util import *


def visualize_groups(dfs_groups, alarms, title, lw=50):
    # Visualize dataset groups

    if DEBUG:
        fig, ax = plt.subplots(figsize=(10, 12))
    else:
        fig, ax = plt.subplots()

    fig_name = 'Classes and Groups'
    ax.set_title(f'{fig_name} {title}', fontsize=15)

    group_color_map = map(lambda dfs_grp: hash(dfs_grp[1]), dfs_groups)

    unique_colors = list(dict.fromkeys(group_color_map))
    unique_groups = list(dict.fromkeys(map(itemgetter(1), dfs_groups)))

    norm = Normalize(vmin=min(group_color_map) - 100, vmax=max(group_color_map) + 100)
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap_data)
    colors = map(lambda c: cmap.to_rgba(c), group_color_map)

    n_samples = len(alarms)
    hits_misses = alarms.loc[map(itemgetter(0), dfs_groups)]['HIT']

    assert len(hits_misses) == n_samples

    ax.scatter(range(n_samples), [3.5] * n_samples,
               c=list(hits_misses), marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(n_samples), [.5] * n_samples,
               c=colors, marker='_', lw=lw, cmap=cmap_data)

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    ax.legend(
        [Patch(color=cmap_cv(.9)), Patch(color=cmap_data(0)), extra] + [Patch(color=cmap.to_rgba(c)) for c in
                                                                        unique_colors] if DEBUG else [],
        ['HIT', 'MISS', ''] + unique_groups if DEBUG else [],
        loc=(1.02, 0.8)
    )

    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['group', 'class'], xlabel="Sample index")
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'{fig_name}.png')


def plot_cv_indices(cv, splits, groups, alarms, title, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    # fig, ax = plt.subplots(figsize=(20, 10))
    fig, ax = plt.subplots()
    n_samples = sum(map(len, splits[0]))
    n_splits = len(splits)
    group_color_map = map(lambda dfs_grp: hash(dfs_grp[1]), groups)

    hits_misses = alarms.loc[map(itemgetter(0), groups)]['HIT']

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

    ax.scatter(range(n_samples), [i + 1.5] * n_samples,
               c=list(hits_misses), marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(n_samples), [i + 2.5] * n_samples,
               c=group_color_map, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 2.2, -.2], xlim=[0, n_samples])
    fig_name = '{}'.format(type(cv).__name__)
    ax.set_title(f'{fig_name} {title}', fontsize=15)

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

    ax.legend(
        [Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02)), extra],
        ['Testing set', 'Training set', f'N groups: {len(set(group_color_map))}'],
        loc=(1.02, .8)
    )
    # Make the legend fit
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'{fig_name}.png')
