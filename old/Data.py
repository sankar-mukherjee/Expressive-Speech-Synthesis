import unicodedata
import re

import torch
from torch.utils.data import Dataset, DataLoader
from old.utils import *


class SpeechDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, r=slice(0, None)):
        'Initialization'
        print('Start loading data')
        fpaths, texts = get_blizzard_data(hp.data, r)
        print('Finish loading data')
        self.fpaths = fpaths
        self.texts = texts

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.fpaths)

    def __getitem__(self, idx):
        _, mel, mag = load_spectrograms(self.fpaths[idx])
        mel = torch.from_numpy(mel)
        mag = torch.from_numpy(mag)
        GO_mel = torch.zeros(1, mel.size(1))  # GO frame
        mel = torch.cat([GO_mel, mel], dim=0)
        text = self.texts[idx]
        return {'text': text, 'mel': mel, 'mag': mag}


def get_blizzard_data(data_dir, r):
    file_list = 'filelists/bliz13_audio_text_train_filelist.txt'

    texts = []
    wav_paths = []
    with open(file_list, 'r') as f:
        for line in f.readlines():
            wav_path, text = line.strip().split('|')
            wav_paths.append(os.path.join(wav_path))

            text = text_normalize(text) + 'E'
            text = [hp.char2idx[c] for c in text]
            text = torch.Tensor(text).type(torch.LongTensor)
            texts.append(text)

    # for wav in wav_paths[-20:]:
    #     print(wav)

    return wav_paths[r], texts[r]


def collate_fn(batch):
    '''
    texts: [N, max_T_x]
    mels:  [N, max_T_y/r, n_mels*r]
    mags:  [N, max_T_y, 1+n_fft/2]
    '''

    texts = [d['text'] for d in batch]
    mels = [d['mel'] for d in batch]
    mags = [d['mag'] for d in batch]

    texts = pad_sequence(texts)
    mels = pad_sequence(mels)
    mags = pad_sequence(mags)

    return {'text': texts, 'mel': mels, 'mag': mags}


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


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


def get_eval_data(text, wav_path):
    '''
    get data for eval
    --------------
    input:
        text --- pinyin format sequence
    output:
        text --- [1, T_x]
        mel ---  [1, 1, n_mels]
    '''
    text = text_normalize(text) + 'E'
    text = [hp.char2idx[c] for c in text]
    text = torch.Tensor(text).type(torch.LongTensor)  # [T_x]
    text = text.unsqueeze(0)  # [1, T_x]
    mel = torch.zeros(1, 1, hp.n_mels)  # GO frame [1, 1, n_mels]

    _, ref_mels, _ = load_spectrograms(wav_path)
    ref_mels = torch.from_numpy(ref_mels).unsqueeze(0)

    return text, mel, ref_mels


if __name__ == '__main__':
    dataset = SpeechDataset(r=slice(hp.eval_size, 15))
    loader = DataLoader(dataset=dataset, batch_size=8, collate_fn=collate_fn)

    for batch in loader:
        print(batch['text'][0])
        print(batch['mel'].size())
        print(batch['mag'].size())
        break
