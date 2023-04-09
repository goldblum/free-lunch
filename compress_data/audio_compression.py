import torch
import numpy as np
import torchaudio
import os
import argparse
from utils import numpy_dataset_size_asarray
import string
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress Librispeech')
    parser.add_argument('--data_root', default='~/data', type=str, help='data directory')
    parser.add_argument('--save_root', default='./', type=str, help='path for saving')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', action='store_true', help='debug mode, fast')
    parser.add_argument('--shuffle', action='store_true', help='shuffle numpy array')
    args = parser.parse_args()
    print(args)

    data_root = args.data_root
    save_root = args.save_root
    if args.debug:
        num_LIBRISPEECH_samples = 2
    else:
        num_LIBRISPEECH_samples = -1

    LIBRISPEECH = torchaudio.datasets.LIBRISPEECH(data_root, download=True)
    data_list = []
    for idx in range(len(LIBRISPEECH)):
        if idx == num_LIBRISPEECH_samples:
            break
        waveform, sample_rate, _, _, _, _ = LIBRISPEECH[idx]
        data_list.append(waveform.numpy().squeeze())
    LIBRISPEECH_data = np.concatenate(data_list)
    if args.shuffle:
        np.random.shuffle(LIBRISPEECH_data)
    print('dataset created')

    raw_bits, pickle_bits, bz2_bits = numpy_dataset_size_asarray(LIBRISPEECH_data, save_root=save_root, bits_per_entry=32)
    
    print('LIBRISPEECH:')
    print('raw_bits', raw_bits)
    print('pickle_bits', pickle_bits)
    print('bz2_bits', bz2_bits)
    suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

    flac_path = os.path.join(save_root, 'temp'+suffix+'.flac')
    torchaudio.save(flac_path, torch.from_numpy(LIBRISPEECH_data).unsqueeze(0), 16000, format='flac')
    flac_bits = os.path.getsize(flac_path)*8
    print('flac_bits', flac_bits)
    os.remove(flac_path)







