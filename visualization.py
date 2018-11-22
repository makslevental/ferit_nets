from operator import itemgetter

from matplotlib.patches import Patch
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

from rewrite import clean_info_df, group_alarms, map, filter


def visualize_groups(classes, groups):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    ax.scatter(range(len(groups)), [.5] * len(groups), c=groups, marker='_',
               lw=50, cmap=cmap_data)
    ax.scatter(range(len(groups)), [3.5] * len(groups), c=classes, marker='_',
               lw=50, cmap=cmap_data)
    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['Data\ngroup', 'Data\nclass'], xlabel="Sample index")
    plt.show()


def plot_cv_indices(cv, X, y, group, n_splits, lw=10):

    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots()
    # Generate the training/testing visualizations for each CV split
    for ii, (train_indices, test_indices) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[test_indices] = 1
        indices[train_indices] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=map(hash, y), marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=map(hash, group), marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 2.2, -.2], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
    # Make the legend fit
    plt.tight_layout()
    plt.show()


df = pd.read_csv('all_maxs.csv')
df = df.where((pd.notnull(df)), None)
df = clean_info_df(df)
groups = group_alarms(df)


n_splits = 10
counts = Counter(groups['group_targetid_map'].values())
group_indices = [idx for idx, group in groups['group_targetid_map'].items() if counts[group] >= n_splits]
groups = [group for _idx, group in groups['group_targetid_map'].items() if counts[group] >= n_splits]

# visualize_groups(list(df.loc[alarm_indices]['HIT']), groups)

cv = StratifiedKFold(n_splits=n_splits)
plot_cv_indices(cv, group_indices, groups, groups, n_splits)

# cv = GroupKFold(n_splits=n_splits)
# plot_cv_indices(cv, alarm_indices, classes, groups, n_splits)
# #
# cv = KFold(n_splits)
# plot_cv_indices(cv, alarm_indices, classes, groups, n_splits)
