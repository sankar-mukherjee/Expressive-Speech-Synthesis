import os
from os.path import join

import torch

from old.Hyperparameters import Hyperparameters as hp
from old.utils import load_spectrograms


def preprocess(data_dir):

    for file in os.listdir(data_dir):
        path = join(data_dir, file)
        _, mel, mag = load_spectrograms(path)
        mel = torch.from_numpy(mel)
        mag = torch.from_numpy(mag)
        GO_mel = torch.zeros(1, mel.size(1))  # GO frame
        mel = torch.cat([GO_mel, mel], dim=0)

    return {'mel': mel, 'mag': mag}

if __name__ == '__main__':
    preprocess(hp.data)
