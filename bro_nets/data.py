import h5py
import os

import torch
# from numba import njit, jit
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from bro_nets.util import profile
from torchvision import transforms
import torch.serialization

from bro_nets.cross_val import tuf_table_csv_to_df, region_and_stratified

from bro_nets import DEBUG


# class Rescale(object):
#     """Rescale the image in a sample to a given size.
#
#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(tuf_self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
#
#         new_h, new_w = int(new_h), int(new_w)
#
#         img = transform.resize(image, (new_h, new_w))
#
#         # h and w are swapped for landmarks because for images,
#         # x and y axes are axis 1 and 0 respectively
#         landmarks = landmarks * [new_w / w, new_h / h]
#
#         return {'image': img, 'landmarks': landmarks}
#

# class RandomCrop(object):
#     """Crop randomly the image in a sample.
#
#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size
#
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
#
#         image = image[top: top + new_h,
#                 left: left + new_w]
#
#         landmarks = landmarks - [left, top]
#
#         return {'image': image, 'landmarks': landmarks}
#

def compute_normalization(dataset: Dataset):
    all_tensors = torch.stack([t for t, c in dataset])
    mean = all_tensors.mean(dim=0)
    std = all_tensors.std(dim=0)
    return mean, std


def compute_normalization_online(dataset: Dataset):
    mk_minus_1 = dataset[0][0]
    sk_minus_1 = torch.zeros_like(mk_minus_1)
    for k, (xk, _) in enumerate(dataset, 1):
        print('norm ', k)
        m = (xk - mk_minus_1)
        mk = mk_minus_1 + m / k
        sk = sk_minus_1 + m * (xk - mk)

        mk_minus_1 = mk
        sk_minus_1 = sk

    return mk, torch.sqrt(sk/(k-1))


class AlarmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root_dir=os.path.join(os.getcwd(), 'data'),
                 feature_type='NNsig_NoInt_15_300', transform=None, mean=None, std=None):
        self.alarms_frame = df.copy(deep=True)
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.feature_type = feature_type
        if mean is not None and std is not None:
            self._mean = mean
            self._std = std
        else:
            # BAD BAD BAD
            self._mean, self._std = compute_normalization_online(self)



    def __len__(self):
        return len(self.alarms_frame)

    def __getitem__(self, idx):
        row = self.alarms_frame.iloc[idx]
        file_name = os.path.join(
            self.data_root_dir,
            row['sample'],
            row['event_id']
        )
        file = h5py.File(f'{file_name}.h5', 'r')

        feature = torch.from_numpy(file[self.feature_type].value)
        if getattr(self, '_mean', None) is not None:
            feature -= self._mean
        if getattr(self, '_std', None) is not None:
            feature /= self._std

        hit = row['HIT']
        # if self.transform:
        #     feature = self.transform(feature)

        return feature, hit


def create_dataloader(alarms: pd.DataFrame, data_root_dir, batch_size=128, shuffle=True, m=None, s=None) -> DataLoader:
    data_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    # m, s = torch.serialization.load('mean.pt'), torch.serialization.load('std.pt')
    al = AlarmDataset(alarms, data_root_dir, mean=m, std=s)

    if DEBUG:
        # https://intellij-support.jetbrains.com/hc/en-us/community/posts/360000066410-pydev-debugger-performing-a-KeyboardInterrupt-while-debugging-program-Error-in-atexit-run-exitfuncs-
        # error pops up in os.fork (or something like that) so don't use multiple workers to avoid
        return DataLoader(al, batch_size, shuffle)
    else:
        return DataLoader(al, batch_size, shuffle, num_workers=4)


if __name__ == '__main__':
    root = os.getcwd()
    # tuf_table_file_name = 'small_maxs_table.csv'
    tuf_table_file_name = 'all_maxs.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(root, tuf_table_file_name))
    # region_and_stratified(all_alarms, n_splits_region=3, n_splits_stratified=10)
    # m1, s1 = compute_normalization_online(al)
    # print(m1, s1)
    # torch.serialization.save(m1, 'means.pt')
    # torch.serialization.save(s1, 'stds.pt')
    # profile(compute_normalization, [al], sort=['cumtime'])
    dl = create_dataloader(all_alarms, os.path.join(root, 'data'), batch_size=32)
