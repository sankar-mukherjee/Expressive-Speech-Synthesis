import torch


class Hyperparameters():

    data = '../database/blizzard2013/segmented/wavn'

    # linguistic
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # english
    char2idx = {char: idx for idx, char in enumerate(vocab)}

    # preprocessing
    sr = 16000  # sampling frequency
    preemphasis = .97  # or None
    n_fft = 1024  # fft points (samples) - ALE changed this from 2048
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples.
    win_length = int(sr * frame_length)  # samples.
    n_mels = 80  # Number of Mel banks to generate
    max_db = 100
    ref_db = 20
    # padding
    r = 5

    # training
    eval_size = 1
    # device select
    use_gpu = torch.cuda.is_available()
    device = 'cuda:0'
    # log
    log_dir = 'log/train{}'


if __name__ == '__main__':
    print(Hyperparameters.char2idx['E'])
