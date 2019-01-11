import os
from typing import Tuple, Union, List

import h5py
import pandas as pd
import torch
import torch.serialization
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tuf_torch.config import DEBUG, DATA_ROOT, PROJECT_ROOT, BATCH_SIZE, SHUFFLE_DL
from tuf_torch.cross_val import tuf_table_csv_to_df


def compute_normalization(dataset: Dataset):
    all_tensors = torch.stack([t for t, c in dataset])
    mean = all_tensors.mean(dim=0)
    std = all_tensors.std(dim=0)
    return mean, std


def compute_normalization_online(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    mk_minus_1 = dataset[0][0]
    sk_minus_1 = torch.zeros_like(mk_minus_1)
    for k, (xk, _) in enumerate(dataset, 1):
        print('norm ', k)
        m = (xk - mk_minus_1)
        mk = mk_minus_1 + m / k
        sk = sk_minus_1 + m * (xk - mk)

        mk_minus_1 = mk
        sk_minus_1 = sk

    return mk, torch.sqrt(sk / (k - 1))


class Normalize(object):
    def __init__(self, mean: torch.Tensor, stddev: torch.Tensor):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return (sample - self.mean) / self.stddev


class AlarmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root_dir: str, feature_type='NNsig_NoInt_15_300', transform=None):
        self.alarms_frame = df.copy(deep=True)
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.feature_type = feature_type

    def __len__(self) -> int:
        return len(self.alarms_frame)

    def __getitem__(self, idx) -> Union[Tuple[torch.Tensor, int], List[Tuple[torch.Tensor, int]]]:
        if isinstance(idx, slice):
            return [self[ix] for ix in range(len(self))[idx]]
        row = self.alarms_frame.iloc[idx]
        file_name = os.path.join(
            self.data_root_dir,
            row['sample'],
            row['event_id']
        )
        file = h5py.File(f'{file_name}.h5', 'r')

        feature = torch.from_numpy(file[self.feature_type].value)
        hit = row['HIT']
        if self.transform:
            feature = self.transform(feature)

        return feature, hit




if __name__ == '__main__':
    tuf_table_file_name = 'small_maxs_table.csv'
    # tuf_table_file_name = 'all_maxs.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, tuf_table_file_name))
    m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
             torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))

    ad = AlarmDataset(
        all_alarms,
        DATA_ROOT,
        transform=transforms.Compose([Normalize(m1, s1)])
    )

    if DEBUG:
        # https://intellij-support.jetbrains.com/hc/en-us/community/posts/360000066410-pydev-debugger-performing-a-KeyboardInterrupt-while-debugging-program-Error-in-atexit-run-exitfuncs-
        # error pops up in os.fork (or something like that) so don't use multiple workers to avoid
        adl = DataLoader(ad, BATCH_SIZE, SHUFFLE_DL)
    else:
        adl = DataLoader(ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)
    print(ad[1:10])
