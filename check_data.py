import os.path
import time
from os.path import join as osj

import numpy as np
from matplotlib import pyplot as plt

from globals import args

split = 'validation'
assert split in ['training', 'validation', 'testing']


def check_data():
    start_chunk = 56
    start_idx = 28
    chunk_idx = start_chunk

    path = osj(args.cache_path, f'cache_{split}', f'batch_{chunk_idx}.npy')
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)

        for idx, btch in enumerate(data[start_idx:]):
            bp_y, bpmean, bpmax = btch['y_bp'], btch['y_phys_mean'], btch['y_phys_max']

            ### Plot ###
            bp = bp_y * bpmax + bpmean
            plt.plot(bp, label='Ground Truth')

            plt.title(f'chunk: {chunk_idx}, idx: {start_idx+idx}')
            plt.legend()
            plt.show()
            time.sleep(2)


if __name__ == '__main__':
    check_data()
