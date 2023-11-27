import json
import os
import random
from glob import glob
from os.path import join as osj

import cv2
import numpy as np
import torch

from dataloader import clean_bvp_signal, post_process_mean_std_vid
from globals import args, tqdm, Logging
from utils import get_all_frames

data_dir = args.datadir
logger = Logging().get(__name__, args.loglevel)

split = 'training'
assert split in ['training', 'validation', 'testing']


def create_cache(split_):
    datapath = osj('..', data_dir, 'V4V', 'V4V_original', 'V4V Dataset')
    if split_ == 'training':
        subjects = glob(osj(datapath, 'Phase 1_ Training_Validation sets', 'Videos', 'Training', '*.mkv'))
    elif split_ == 'validation':
        subjects = glob(osj(datapath, 'Phase 1_ Training_Validation sets', 'Videos', 'Validation', '*.mkv'))
    elif split_ == 'testing':
        subjects = glob(osj(datapath, 'Phase 2_ Testing set', 'Videos', 'Test', 'test', '*.mkv'))
    else:
        raise NotImplementedError()

    lens, seqs = len(subjects), []

    subject_list = sorted(subjects)
    random.seed(123)
    if split_ == 'training':
        random.shuffle(subject_list)

    for vidname in tqdm(subject_list):
        cap = cv2.VideoCapture(vidname)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if length < args.seqlen:
            logger.debug(f'Subject {vidname} has < {args.seqlen} frames with {length} frames')
            continue

        for i in range(length // args.seqlen):
            seqs.append((vidname, i * args.seqlen))
            logger.debug(f'{vidname}: {i * args.seqlen}')

    batch_size = 50
    num_batches = (len(seqs) - 1) // batch_size + 1

    for batch_index in range(num_batches):
        start_idx = batch_index * batch_size
        end_idx = start_idx + batch_size

        batch_data = seqs[start_idx:end_idx]
        batch_caches = []

        for i, (vidpath, frameidx) in enumerate(batch_data):

            subject = vidpath.split('/')[-1].split('.')[0]

            frames = get_all_frames(vidpath, frameidx, frameidx + args.seqlen, (36, 36))
            assert len(
                frames) == args.seqlen, f'Turns out {len(frames)}=={args.seqlen} for {subject} with {frameidx}'
            dataX = np.stack(frames)

            subjbpname = subject.replace('_', '-')
            if split_ == 'training':
                subjpath = osj(datapath, 'Phase 1_ Training_Validation sets', 'Ground truth', 'BP_raw_1KHz',
                               f'{subjbpname}-BP.txt')
            if split_ == 'validation':
                subjpath = osj(datapath, 'Phase 2_ Testing set', 'blood_pressure', 'val_set_bp',
                               f'{subjbpname}.txt')
            if split_ == 'testing':
                subjpath = osj(datapath, 'Phase 2_ Testing set', 'blood_pressure', 'test_set_bp',
                               f'{subjbpname}.txt')

            y_phys, y_phys_mean, y_phys_max = clean_bvp_signal(subjpath, frameidx, args.seqlen, subject)

            assert len(dataX) == len(y_phys), ['Length validation',
                                               f'{subject}, {frameidx}:: {len(dataX)}--{len(y_phys)}']

            retdict = {'X': dataX, 'y_bp': y_phys, 'y_phys_mean': y_phys_mean, 'y_phys_max': y_phys_max, 'y_hr': torch.tensor([])}

            debug_dict = {'subject': subject, 'frameidx': frameidx}

            ####

            debug_dict['oriX'] = np.copy(retdict['X'])
            retdict['X'] = post_process_mean_std_vid(args.seqlen, retdict['X'])

            assert len(retdict['y_bp']) == len(retdict['X'])

            batch_caches.append(retdict)

        root_directory = f'cache_{split_}'
        if not os.path.exists(root_directory):
            os.mkdir(root_directory)

        batch_cache_filename = osj(root_directory, f'batch_{batch_index}.npy')
        np.save(batch_cache_filename, batch_caches)


def create_cache_meta(split_):
    datapath = osj('..', data_dir, 'V4V', 'V4V_original', 'V4V Dataset')
    if split_ == 'training':
        subjects = glob(osj(datapath, 'Phase 1_ Training_Validation sets', 'Videos', 'Training', '*.mkv'))
    elif split_ == 'validation':
        subjects = glob(osj(datapath, 'Phase 1_ Training_Validation sets', 'Videos', 'Validation', '*.mkv'))
    elif split_ == 'testing':
        subjects = glob(osj(datapath, 'Phase 2_ Testing set', 'Videos', 'Test', 'test', '*.mkv'))
    else:
        raise NotImplementedError()

    lens, seqs = len(subjects), []

    subject_list = sorted(subjects)
    random.seed(123)
    if split_ == 'training':
        random.shuffle(subject_list)

    for vidname in tqdm(subject_list):
        cap = cv2.VideoCapture(vidname)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if length < args.seqlen:
            logger.debug(f'Subject {vidname} has < {args.seqlen} frames with {length} frames')
            continue

        for i in range(length // args.seqlen):
            seqs.append((vidname, i * args.seqlen))
            logger.debug(f'{vidname}: {i * args.seqlen}')

    batch_size = 50
    num_batches = (len(seqs) - 1) // batch_size + 1

    for batch_index in range(num_batches):
        start_idx = batch_index * batch_size
        end_idx = start_idx + batch_size

        batch_data = seqs[start_idx:end_idx]

        root_directory = f'cache_{split_}'
        if not os.path.exists(root_directory):
            os.mkdir(root_directory)

        batch_metadata = {'num_items': len(batch_data), 'sequences': batch_data}
        metadata_filename = osj(root_directory, f'batch_{batch_index}_metadata.json')
        with open(metadata_filename, 'w') as metadata_file:
            json.dump(batch_metadata, metadata_file, sort_keys=True, indent=4)


if __name__ == '__main__':
    create_cache_meta(split)
