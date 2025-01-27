import random
from collections import OrderedDict

import pandas as pd
from numpy import split

from globals import *
logger = Logging().get(__name__, args.loglevel)

from torch.utils.data import DataLoader
from utils import get_all_frames, detrend_filter

torch.multiprocessing.set_sharing_strategy('file_system')


def clean_bvp_signal(path_file, frameidx, seqlen, subject):
    phys_1k = np.loadtxt(path_file, delimiter='\n')
    phys = np.array([np.mean(phys_1k[i:i + 40]) for i in range(0, len(phys_1k), 40)])
    phys = phys[frameidx: frameidx + seqlen]
    assert len(phys) > 0, ['Missing Phys', subject]
    # phys = detrend_filter(phys, cumsum=False)

    # y_phys = norm_sig(phys)
    phys_mean = phys.mean()
    phys_mean_centered = phys - phys_mean
    phys_mean_centered_absolute_max = np.max(np.abs(phys_mean_centered))
    phys_scaled = phys_mean_centered / phys_mean_centered_absolute_max
    y_phys = np.reshape(phys_scaled, -1)
    return y_phys, phys_mean, phys_mean_centered_absolute_max


def post_process_mean_std_vid(seqlen, seqX):
    seqOri = np.copy(seqX)
    for i in range(seqlen - 1):
        diff = (seqX[i + 1] - seqX[i]) / (seqX[i + 1] + seqX[i] + 3e-5)
        seqX[i] = diff

    seqX[-1] = seqX[-2]
    seqX = seqX / (np.std(seqX) + 3e-5)
    assert seqX.shape == seqOri.shape

    seqOri = seqOri - np.mean(seqOri)
    seqOri = seqOri / np.std(seqOri)

    seqX = np.concatenate([seqX, seqOri], -1)

    return seqX


def post_process_signal_first_derivative(seqlen, sig):
    sigdiff = np.copy(sig)
    for i in range(seqlen - 1):
        sigdiff[i] = (sig[i + 1] - sig[i])

    sigdiff[-1] = sigdiff[-2]
    sigdiff = sigdiff / max(sigdiff)

    return sigdiff


def get_appearance_motion(image_batch):
    assert image_batch.shape[1:] == torch.Size([3, 36, 36]) and len(image_batch.shape) == 4

    lshifted = torch.cat([image_batch, torch.zeros(1, 3, 36, 36).cuda()], 0)[1:]
    motion = (image_batch / 2 - lshifted / 2) / (image_batch / 2 + lshifted / 2 + 1e-5)
    motion[motion > 3] = 3
    motion[motion < -3] = -3
    motion = motion[:-1]
    motion = (motion - motion.mean()) / motion.std()

    appearance = image_batch[:-1]
    estimated_mean, estimated_std = torch.mean(appearance), torch.std(appearance)
    appearance = (appearance - estimated_mean) / estimated_std

    return appearance, motion


class V4V_Dataset(DataLoader):
    def __init__(self, split, use_cache=False):
        assert split in ['training', 'validation', 'testing'], ['split issue', 'training']
        self.seqlen = args.seqlen
        self.datapath = osj(data_dir, 'V4V', 'V4V_original', 'V4V Dataset')
        self.use_cache = use_cache
        self.split = split

        self.gt_validation = {}
        self.gt_test = {}

        logger.info(f'V4V Dataset with split: {self.split}')

        self.data_seq_list = self.__config()

        self.cachepath = f'cache_{self.split}_{args.subset_size}.npy'
        # Load cached data if available
        if self.use_cache and os.path.exists(self.cachepath):
            self.cached_data = np.load(self.cachepath, allow_pickle=True).item()
        else:
            self.cached_data = {}  # Initialize an empty cache

    def __config(self):
        if self.split == 'training':
            subjects = glob(osj(self.datapath, 'Phase 1_ Training_Validation sets', 'Videos', 'Training', '*.mkv'))
        elif self.split == 'validation':
            subjects = glob(osj(self.datapath, 'Phase 1_ Training_Validation sets', 'Videos', 'Validation', '*.mkv'))
        elif self.split == 'testing':
            subjects = glob(osj(self.datapath, 'Phase 2_ Testing set', 'Videos', 'Test', 'test', '*.mkv'))
        else:
            raise NotImplementedError()

        lens, seqs = len(subjects), []

        subject_list = sorted(subjects)
        random.seed(123)
        if self.split == 'training':
            random.shuffle(subject_list)

        if args.use_subset:
            subject_list = random.sample(subject_list, args.subset_size)

        for vidname in tqdm(subject_list):
            subjname = vidname.split('/')[-1]
            ### ### ###
            cap = cv2.VideoCapture(vidname)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ### ### ###

            if length < self.seqlen:
                logger.debug(f'Subject {vidname} has < {self.seqlen} frames with {length} frames')
                continue

            for i in range(length // self.seqlen):
                seqs.append((vidname, i * self.seqlen))
                logger.debug(f'{vidname}: {i * self.seqlen}')

        if self.split == 'validation' or self.split == 'testing':
            fname = 'validation_set_gt_release.txt' if self.split == 'validation' else 'test_set_gt_release.txt'
            gtfile = osj(self.datapath, 'Phase 2_ Testing set', fname)
            with open(gtfile) as f:
                start_idx = 0 if PHYS_TYPE == 'HR' else 1
                for line in f.readlines()[start_idx::2]:
                    line = line.split(',')
                    subj, phys, hr = line[0].strip(), line[1].strip(), np.array(line[2:], dtype=np.float32)
                    assert phys == PHYS_TYPE
                    if self.split == 'validation':
                        self.gt_validation[subj] = hr.astype(np.float32)
                    elif self.split == 'testing':
                        self.gt_test[subj] = hr.astype(np.float32)

        return seqs

    def postprocess(self, datadict, subject, frameidx):
        debug_dict = {'subject': subject, 'frameidx': frameidx}

        ####

        debug_dict['oriX'] = np.copy(datadict['X'])
        datadict['X'] = post_process_mean_std_vid(self.seqlen, datadict['X'])
        # datadict['y_bp'] = post_process_signal_first_derivative(self.seqlen, datadict['y_bp'])

        assert len(datadict['y_bp']) == len(datadict['X'])
        return datadict, debug_dict

    def __getitem__(self, idx):
        # Check if data is already cached for this index
        if self.use_cache and idx in self.cached_data:
            return self.cached_data[idx]
        vidpath, frameidx = self.data_seq_list[idx]

        subject = vidpath.split('/')[-1].split('.')[0]

        #### 2. Video frames
        subjname = vidpath.split('/')[-1]
        subj = subjname.split('.')[0]

        frames = get_all_frames(vidpath, frameidx, frameidx + self.seqlen, (36, 36))
        assert len(frames) == self.seqlen, f'Turns out {len(frames)}=={self.seqlen} for {subj} with {frameidx}'
        dataX = np.stack(frames)

        if PHYS_TYPE == 'HR':
            #### 3. BVP signal with butterworth
            subjbpname = subject.replace('_', '-')
            if self.split == 'training':
                subjpath = osj(self.datapath, 'Phase 1_ Training_Validation sets', 'Ground truth', 'BP_raw_1KHz',
                               f'{subjbpname}-BP.txt')
            if self.split == 'validation':
                subjpath = osj(self.datapath, 'Phase 2_ Testing set', 'blood_pressure', 'val_set_bp',
                               f'{subjbpname}.txt')
            if self.split == 'testing':
                subjpath = osj(self.datapath, 'Phase 2_ Testing set', 'blood_pressure', 'test_set_bp',
                               f'{subjbpname}.txt')

            # if self.split in ['training', 'validation']:
            y_phys, y_phys_mean, y_phys_max = clean_bvp_signal(subjpath, frameidx, self.seqlen, subject)
            # else:
            #     y_phys = np.zeros(dataX.shape[0])

            #### 4. Heart rate
            if self.split == 'training':
                hr = np.loadtxt(osj(self.datapath, 'Phase 1_ Training_Validation sets', 'Ground truth', 'Physiology',
                                    f'{subject}.txt'), delimiter=',', dtype=str)
                hr = hr[0, 2:].astype(np.float32)
                hr = hr[frameidx:frameidx + self.seqlen]
            elif self.split == 'validation':
                hr = self.gt_validation[f'{subject}.mkv'][frameidx:frameidx + self.seqlen]
            elif self.split == 'testing':
                hr = self.gt_test[f'{subject}.mkv'][frameidx:frameidx + self.seqlen]
            else:
                raise NotImplementedError()
        else:
            #### 3. RR signal with butterworth
            subj_head, subj_tail = subject.split('_')
            if self.split in ['training', 'validation']:
                subjpath = osj(data_dir, 'bp4dPhys', subj_head, subj_tail, 'Respiration Rate_BPM.txt')
                y_phys = clean_bvp_signal(subjpath, frameidx, self.seqlen, subject)
            else:
                y_phys = np.zeros(dataX.shape[0])

            #### 4. Respiration rate -- Variable name has been reused.
            if self.split == 'training':
                hr = np.loadtxt(osj(self.datapath, 'Training', 'Physiology', f'{subject}.txt'), delimiter=',',
                                dtype=str)
                hr = hr[1, 2:].astype(np.float32)
                hr = hr[frameidx:frameidx + self.seqlen]
            elif self.split == 'validation':
                hr = self.gt_validation[f'{subject}.mkv'][frameidx:frameidx + self.seqlen]
            elif self.split == 'testing':
                hr = self.gt_test[f'{subject}.mkv'][frameidx:frameidx + self.seqlen]
            else:
                raise NotImplementedError()

        #### 5. Length
        assert len(dataX) == len(y_phys) and len(dataX) == len(hr), \
            ['Length validation', f'{subject} :: {len(dataX)}--{len(y_phys)}--{len(hr)}']

        retdict = {'X': dataX, 'y_bp': y_phys, 'y_phys_mean': y_phys_mean, 'y_phys_max': y_phys_max, 'y_hr': hr}
        postprocess_retdict = self.postprocess(retdict, subject, frameidx)

        # Cache the data for this index
        if self.use_cache:
            self.cached_data[idx] = postprocess_retdict
            np.save(self.cachepath, self.cached_data)  # Save the updated cache

        return postprocess_retdict

    def __len__(self):
        return len(self.data_seq_list)


class DynamicV4VDataset(DataLoader):
    def __init__(self, split):

        self.allowed_classes = [0]
        self.cache = OrderedDict()

        assert split in ['training', 'validation', 'testing'], ['split issue', 'training']
        self.chunk_size = 50
        self.split = split

        # Cache management parameters
        biggest_chunk = 317  # MB
        max_cache_size = 16_000  # MB
        self.max_number_of_chunks = max_cache_size // biggest_chunk

        ######## Mask ########
        # TODO
        mask = self.create_mask()
        self.masked_cache = mask[mask.isin(self.allowed_classes)].stack().index
        self.total_length = len(self.masked_cache)

        logger.info(f'V4V Dataset with split: {self.split}')

    def __getitem__(self, idx):
        row_idx, column_idx = self.masked_cache[idx]
        column_idx = int(column_idx)
        data = self.load_batch(row_idx)[column_idx]
        return data, 0 # TODO remove second return

    def __len__(self):
        if args.use_subset:
            # TODO logging
            return min(args.subset_size, self.total_length)
        return self.total_length

    def create_mask(self):
        mask_file_path = osj(args.cache_mask_path, f'{self.split}_data_mask.csv')
        if os.path.exists(mask_file_path):
            return pd.read_csv(mask_file_path, index_col=0)
        else:
            raise Exception(f'Data not found.')

    def load_batch(self, chunk_idx):
        if chunk_idx in self.cache:
            self.cache.move_to_end(chunk_idx)
            return self.cache[chunk_idx]

        path = osj(args.cache_path, f'cache_{self.split}', f'batch_{chunk_idx}.npy')
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            if len(self.cache) + 1 > self.max_number_of_chunks:
                self.cache.popitem(last=False)
            self.cache[chunk_idx] = data
            return data
        else:
            raise Exception(f'Data not found chunk {chunk_idx}')
