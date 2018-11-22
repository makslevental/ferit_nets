import os
from collections import Counter
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import StratifiedKFold, KFold
import h5py
import numpy as np
import pandas as pd
import pyproj
from operator import itemgetter
from torch.utils.data import Dataset

old_filter = filter
filter = lambda x, y: list(old_filter(x, y))
old_zip = zip
zip = lambda x, y: list(old_zip(x, y))
old_map = map
map = lambda x, y: list(old_map(x, y))


class AlarmDataset(Dataset):
    def __init__(self, csv_file, root_dir, feature_type='NNsig_NoInt_15_300', transform=None):
        self.alarms_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.feature_type = feature_type

    def __len__(self):
        return len(self.alarms_frame)

    def __getitem__(self, idx):
        row = self.alarms_frame.iloc[idx]
        file_name = os.path.join(
            self.root_dir,
            row['sample'],
            row['event_id']
        )
        file = h5py.File(f'{file_name}.h5', 'r')

        feature = np.array(file[self.feature_type])
        hit = row['HIT']
        if self.transform:
            feature = self.transform(feature)

        return feature, hit


def clean_info_df(df: pd.DataFrame) -> pd.DataFrame:
    def corners_to_tuple(corners: pd.DataFrame) -> Optional[List[Tuple]]:
        corners = filter(None, corners)
        num_corners = len(corners)
        corner_tuples = tuple(zip(corners[:num_corners // 2], corners[num_corners // 2:]))
        return corner_tuples if len(corner_tuples) else None

    corners_names = filter(lambda x: 'corner' in x, df.columns)
    df['corners'] = df[corners_names].apply(corners_to_tuple, axis=1)
    df.drop(corners_names, inplace=True, axis=1)

    df.drop(
        [
            'xUTM', 'yUTM', 'sensor_type', 'dt', 'ch', 'scan', 'time', 'MineIsInsideLane', 'DIST', 'objectdist',
            'category', 'isDetected', 'IsInsideLane', 'prescreener', 'known_obj', 'event_type', 'id', 'lsd_kv',
            'encoder', 'event_timestamp', 'fold'
        ],
        inplace=True,
        axis=1
    )

    return df


def group_alarms(df: pd.DataFrame):
    grouped = df.dropna(subset=['corners']).groupby(['corners', 'depth', 'target', 'srid'])
    groups = list(grouped.groups.items())

    # match groups with their target (in order to stratify)
    group_targetid_map = {i: group[2] for i, (group, df_indices) in enumerate(groups)}

    return {
        'groups': groups,
        'group_targetid_map': group_targetid_map
    }


def create_cross_val_folds(n_folds: int, groupings: Dict, df: pd.DataFrame):
    skf = StratifiedKFold(n_splits=n_folds)
    kf = KFold(n_splits=n_folds)

    misses = np.random.permutation(df[df['HIT'] == 0].index)

    counts = Counter(groupings['group_targetid_map'].values())
    # if number of instances/grouping is less than the number of folds then you can't distribute across the folds evenly
    not_enough_group = np.array_split(
        np.random.permutation(np.asarray(
            [idx for idx, group in groupings['group_targetid_map'].items() if counts[group] < n_folds],
            dtype=np.dtype("int")
        )),
        n_folds
    )

    enough_group = np.random.permutation(
        [(idx, group) for idx, group in groupings['group_targetid_map'].items() if counts[group] >= n_folds])

    skf_iter = skf.split(
        X=map(itemgetter(0), enough_group),
        y=map(itemgetter(1), enough_group)
    )
    kf_iter = kf.split(X=misses)

    for i in range(n_folds):
        enough_group_train_ids, enough_group_test_ids = next(skf_iter)
        enough_group_train_indices = np.hstack(map(lambda id: groupings['groups'][id][1], enough_group_train_ids))
        enough_group_test_indices = np.hstack(map(lambda id: groupings['groups'][id][1], enough_group_test_ids))

        not_enough_group_train_ids = np.hstack(not_enough_group[:i] + not_enough_group[i + 1:])
        not_enough_group_test_ids = not_enough_group[i]
        not_enough_group_train_indices = np.hstack(
            map(lambda id: groupings['groups'][id][1], not_enough_group_train_ids))
        not_enough_group_test_indices = np.hstack(map(lambda id: groupings['groups'][id][1], not_enough_group_test_ids))

        miss_train_indices, miss_test_indices = next(kf_iter)

        yield (
            np.hstack([enough_group_train_indices, miss_train_indices, not_enough_group_train_indices]),
            np.hstack([enough_group_test_indices, miss_test_indices, not_enough_group_test_indices]),
        )


# dataset = AlarmDataset(
#     csv_file='data/three_region_cross_val.csv',
#     root_dir='data/',
#     transform=transforms.ToTensor()
# )
#
# for train_idx, validation_idx in KFold(n_splits=10).split(np.zeros((len(dataset), 1))):  # (samples, features)
#     train_sampler = SubsetRandomSampler(train_idx)
#     validation_sampler = SubsetRandomSampler(validation_idx)
#
#     train_loader = DataLoader(
#         dataset,
#         batch_size=1,
#         sampler=train_sampler
#     )
#
#     validation_loader = DataLoader(
#         dataset,
#         batch_size=2,
#         sampler=validation_sampler
#     )
#     for epoch in range(2):
#         for batch_index, (inputs, label) in enumerate(train_loader):
#             print(epoch, batch_index, label)
#
#     for epoch in range(2):
#         for batch_index, (inputs, labels) in enumerate(validation_loader):
#             print(epoch, batch_index, labels)

def convert_utm_to_lat_lon(df):
    # Define the two projections.
    # p1 = pyproj.Proj(init='eps:32618')
    p1 = pyproj.Proj('+proj=utm +zone=18 +datum=WGS84 +units=m +no_defs')
    # p2 = pyproj.Proj(init='eps:32601')
    p2 = pyproj.Proj('+proj=utm +zone=1 +datum=WGS84 +units=m +no_defs')
    # p3 = pyproj.Proj(init='eps:32611')
    p3 = pyproj.Proj('+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs')

    df


df = pd.read_csv('all_maxs.csv')
df = df.where((pd.notnull(df)), None)
df = clean_info_df(df)
groups = group_alarms(df)
for train, test in create_cross_val_folds(10, groups, df):
    print(train, test, len(train), len(test))
