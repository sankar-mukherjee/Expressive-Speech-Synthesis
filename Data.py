import os

import torch
from Hyperparameters import Hyperparameters as hp
from torch.utils.data import Dataset, DataLoader
from utils import *


class SpeechDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, r=slice(0, None)):
        'Initialization'
        print('Start loading data')
        fpaths, labels = get_blizzard_data(hp.data, r)
        print('Finish loading data')
        self.fpaths = fpaths
        self.labels = labels

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.fpaths)

    def __getitem__(self, idx):
        _, mel, mag = load_spectrograms(self.fpaths[idx])
        mel = torch.from_numpy(mel)
        mag = torch.from_numpy(mag)
        GO_mel = torch.zeros(1, mel.size(1))  # GO frame
        mel = torch.cat([GO_mel, mel], dim=0)
        return {'labels': self.labels[idx], 'mel': mel, 'mag': mag}


def get_blizzard_data(data_dir, r):
    file_list = os.listdir(data_dir)
    wav_paths = []
    labels = []
    for f in file_list:
        wav_paths.append(os.path.join(data_dir, f))
        labels.append(f)

    # for wav in wav_paths[-20:]:
    #     print(wav)

    return wav_paths[r], labels[r]


def collate_fn(batch):
    '''
    texts: [N, max_T_x]
    mels:  [N, max_T_y/r, n_mels*r]
    mags:  [N, max_T_y, 1+n_fft/2]
    '''

    #labels = [d['labels'] for d in batch]
    mels = [d['mel'] for d in batch]
    mags = [d['mag'] for d in batch]

    #labels = pad_sequence(labels)
    mels = pad_sequence(mels)
    mags = pad_sequence(mags)

    return {'mel': mels, 'mag': mags}


def pad_sequence(sequences):
    '''
    pad sequence to same length (max length)
    ------------------
    input:
        sequences --- a list of tensor with variable length
        out --- a tensor with max length
    '''
    lengths = [data.size(0) for data in sequences]
    batch_size = len(sequences)
    max_len = max(lengths)
    trailing_dims = sequences[0].size()[1:]
    out_dims = (batch_size, max_len) + trailing_dims
    dtype = sequences[0].data.type()
    out = torch.zeros(*out_dims).type(dtype)
    for i, data in enumerate(sequences):
        out[i, :lengths[i]] = data

    return out


if __name__ == '__main__':
    dataset = SpeechDataset(r=slice(hp.eval_size, 16))
    loader = DataLoader(dataset=dataset, batch_size=8, collate_fn=collate_fn)

    for batch in loader:
        #print(batch['labels'][0])
        print(batch['mel'].size())
        print(batch['mag'].size())
        break
