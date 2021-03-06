from datetime import datetime
from operator import itemgetter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from tensorboardX import SummaryWriter

from tuf_torch import ROC

mpl.rcParams['figure.dpi'] = 300

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

now = datetime.now()

writer = SummaryWriter(f'logs/{now}/')


def visualize_groups(dfs_groups, alarms, title: str, lw=50, save=False):
    # Visualize dataset groups

    fig, ax = plt.subplots(figsize=(10, 6))

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
        [
            Patch(color=cmap_cv(.9)),
            Patch(color=cmap_data(0)),
            extra,
            *[Patch(color=cmap.to_rgba(c)) for c in unique_colors]
        ],
        [
            'HIT',
            'MISS',
            '',
            *unique_groups
        ],
        loc=(1.02, 0)
    )

    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['group', 'class'], xlabel="Sample index")
    # plt.tight_layout()
    if save:
        plt.savefig(f'{fig_name}.png')
    else:
        plt.show()
    plt.close(fig)


def plot_roc(roc: ROC, title: str="", label: str="", show=False, save=False, fig: plt.Figure = None):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
    else:
        ax = fig.get_axes()
        assert len(ax) == 1
        ax = ax[0]

    lw = 2
    fpr, tpr, _ = roc
    ax.plot(fpr, tpr, lw=lw, label=label)

    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set(xlabel='FPR', ylabel='TPR',
           ylim=[0.0, 1.05], xlim=[0.0, 1.0])
    ax.set_title(title, fontsize=15)
    ax.legend(loc="lower right")
    # fig.tight_layout()
    if show:
        fig.show()
    if save:
        fig.savefig(f'figs/{title}_{label}.png')
    return fig


def plot_cv_indices(splits, dfs_groups, alarms, title, lw=10, save=False):
    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots(figsize=(10, 6))

    group_color_map = map(lambda dfs_grp: hash(dfs_grp[1]), dfs_groups)
    unique_colors = list(dict.fromkeys(group_color_map))
    unique_groups = list(dict.fromkeys(map(itemgetter(1), dfs_groups)))

    norm = Normalize(vmin=min(group_color_map) - 100, vmax=max(group_color_map) + 100)
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap_data)

    n_samples = sum(map(len, splits[0]))
    n_splits = len(splits)

    hits_misses = alarms.loc[map(itemgetter(0), dfs_groups)]['HIT']

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
    ax.set_title(title, fontsize=15)

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

    ax.legend(
        [
            Patch(color=cmap_cv(.8)),
            Patch(color=cmap_cv(.02)),
            extra,
            *[Patch(color=cmap.to_rgba(c)) for c in unique_colors]
        ],
        [
            'Testing set',
            'Training set',
            f'N groups: {len(set(group_color_map))}',
            *unique_groups
        ],
        loc=(1.02, 0)
    )
    # Make the legend fit
    # plt.tight_layout()
    if save:
        plt.savefig(f'{title}.png')
    else:
        plt.show()
    plt.close(fig)


def plot_auc_loss(log_df, title):
    fig, ax1 = plt.subplots()
    fig.suptitle(title, fontsize=16)
    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('auc', color=color)
    ax1.plot(log_df['strat_auc'].values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(log_df['strat_loss'].values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(top=0.9)
    return fig
