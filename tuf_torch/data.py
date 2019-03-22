import glob
import os
from typing import Tuple, Union, List

import h5py
import pandas as pd
import torch
import torch.serialization
from torch.utils.data import Dataset

import tuf_torch.models427


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

        feature = torch.from_numpy(file[self.feature_type][()])
        hit = row['HIT']
        if self.transform:
            feature = self.transform(feature)

        return feature, hit


def load_nets_dir(dir_path, net_name):
    nets = []
    for p in glob.glob(os.path.join(dir_path, "*.net")):
        state_dict = torch.load(p)
        model = torch.nn.DataParallel(getattr(tuf_torch.models427, net_name)())
        model.load_state_dict(state_dict)
        model.eval()
        nets.append(model)
    return nets


if __name__ == '__main__':
    # tuf_table_file_name = 'all_maxs.csv'
    # # tuf_table_file_name = 'all_maxs.csv'
    # all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, tuf_table_file_name))
    # # m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
    # #          torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))
    # 
    # ad = AlarmDataset(
    #     all_alarms,
    #     DATA_ROOT,
    #     # transform=transforms.Compose([Normalize(m1, s1)])
    # )
    # 
    # m1, s1 = compute_normalization_online(ad)
    # torch.serialization.save(m1, os.path.join(PROJECT_ROOT, 'means.pt'))
    # torch.serialization.save(s1, os.path.join(PROJECT_ROOT, 'stds.pt'))
    nets = load_nets_dir("/home/maksim/ferit_nets/nets/cynicism's_2019-02-21", "GPR_15_300")
    print(nets)
