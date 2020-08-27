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

    # griffin-lin
    n_iter = 50  # Number of inversion iterations

    # boost some frequencies in loss function
    n_priority_freq = int(3000 / (sr * 0.5) * (n_fft / 2))

    # padding
    r = 5

    # training
    eval_size = 1
    batch_size = 8   # !!!
    num_epochs = 100  # !!!
    log_per_batch = 20
    save_per_epoch = 1

    # learning rate
    lr = 0.001
    # change learning rate after steps
    lr_step = [500000, 1000000, 2000000]

    max_Ty = max_iter = 200

    # text embeddings
    E = 256

    # reference encoder
    ref_enc_filters = [32, 32, 64, 64, 128, 128]

    # encoder parameters
    K = 16
    decoder_K = 8
    num_highways = 4
    dropout_p = 0.5

    # style token layer
    token_num = 10
    num_heads = 8

    # device select
    use_gpu = torch.cuda.is_available()
    device = 'cuda:0'
    # log
    log_dir = 'log/train{}'

    eval_text = 'it took me a long time to develop a brain . now that i have it i\'m not going to be silent !'
    ref_wav = '../database/blizzard2013/segmented/wavn/CA-MP2-11-028.wav'


if __name__ == '__main__':
    print(Hyperparameters.char2idx['E'])
