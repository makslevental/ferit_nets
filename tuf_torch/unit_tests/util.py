import csv
import os.path as osp
import random

import pandas as pd

from tuf_torch.config import PROJECT_ROOT


def anonymize_data(csv_fp):
    smallcsv = pd.read_csv(csv_fp, delimiter=r"\s*,", engine='python')

    smallcsv['time'] = smallcsv['time'].map(lambda x: x * random.random())
    smallcsv['prescreener'] = smallcsv['prescreener'].map(lambda x: "random_prescreener")

    anon_lsd_kvs = {lsd_kv: f"anon_lsd_kv_{n}" for n, lsd_kv in enumerate(smallcsv['lsd_kv'].unique())}
    anon_lanes = {lane: f"anon_lane_{n}" for n, lane in enumerate(smallcsv['lane'].unique())}
    anon_sites = {site: f"anon_site_{n}" for n, site in enumerate(smallcsv['site'].unique())}
    anon_targets = {target: f"anon_target_{n}" for n, target in enumerate(smallcsv['target'].unique())}
    smallcsv['lsd_kv'] = smallcsv['sample'] = smallcsv['lsd_kv'].map(lambda l: anon_lsd_kvs[l])
    smallcsv['lane'] = smallcsv['sample'] = smallcsv['lane'].map(lambda l: anon_lanes[l])
    smallcsv['site'] = smallcsv['sample'] = smallcsv['site'].map(lambda l: anon_sites[l])
    smallcsv['target'] = smallcsv['sample'] = smallcsv['target'].map(lambda l: anon_targets[l])

    def shift_corners(row):
        shift = random.randint(100, 200)
        return row + [shift] * len(row)

    smallcsv[['xUTM', 'yUTM', 'corner_1', 'corner_2', 'corner_3', 'corner_4', 'corner_5', 'corner_6']].apply(
        shift_corners, axis=1)
    smallcsv.to_csv(osp.splitext(csv_fp)[0] + "_anon.csv", index=False)

    event_id_map = zip(
        smallcsv['event_id'],
        smallcsv['xUTM'].map(lambda x: f"{int(x)}E_") + smallcsv['yUTM'].map(lambda y: f"{int(y)}NGPR")
    )
    with open(osp.splitext(csv_fp)[0] + "_anon_event_id_map.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['anon', 'true'])
        for true, anon in event_id_map:
            csvwriter.writerow([anon, true])


if __name__ == '__main__':
    tuf_table_file_name = 'small_maxs_table_anon.csv'
    anonymize_data(osp.join(PROJECT_ROOT, tuf_table_file_name))
