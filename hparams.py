
# CONFIG -----------------------------------------------------------------------------------------------------------#

# Here are the input and output data paths (Note: you can override wav_path in preprocess.py)
wav_path = '../database/blizzard2013/segmented/small_wavn_lead_trail_silence_removed_16000'
data_path = 'data_blizzard2013_16000/'

# model ids are separate - that way you can use a new tts with an old wavernn and vice versa
# NB: expect undefined behaviour if models were trained on different DSP settings
voc_model_id = 'blizzard2013_mol'
tts_model_id = 'blizzard2013_lsa_smooth_attention_pytorch_multihead'

# set this to True if you are only interested in WaveRNN
ignore_tts = False


# DSP --------------------------------------------------------------------------------------------------------------#

# Settings for all models
sample_rate = 16000
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 200                    # 12.5ms - in line with Tacotron 2 paper
win_length = 800                   # 50ms - same reason as above
fmin = 40
min_level_db = -100
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False                   # Normalise to the peak of each wav file


# WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#


# Model Hparams
voc_mode = 'MOL'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_upsample_factors = (5, 5, 8)   # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 64
voc_lr = 1e-4
voc_checkpoint_every = 25_000
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_total_steps = 2_000_000         # Total number of training steps
voc_test_samples = 50               # How many unseen samples to put aside for testing
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length

# Generating / Synthesizing
voc_gen_batched = True         # very fast (realtime+) single utterance batched generation
voc_target = 11_000                 # target number of samples to be generated in each batch entry
voc_overlap = 550                   # number of samples for crossfading between batches


# TACOTRON/TTS -----------------------------------------------------------------------------------------------------#


# Model Hparams
tts_r = 1                           # model predicts r frames per output step
tts_embed_dims = 256                # embedding dimension for the graphemes/phoneme inputs
tts_encoder_dims = 128
tts_decoder_dims = 256
tts_postnet_dims = 128
tts_encoder_K = 16
tts_lstm_dims = 512
tts_postnet_K = 8
tts_num_highways = 4
tts_dropout = 0.5
tts_cleaner_names = ['english_cleaners']

# Training


tts_schedule = [(7,  1e-3,  10_000,  8),   # progressive training schedule
                (5,  1e-4, 100_000,  8),   # (r, lr, step, batch_size)
                (2,  1e-4, 180_000,  8),
                (1,  1e-4, 350_000,  8),
                (1,  1e-4, 1000_000,  8)]

tts_max_mel_len = 1250              # if you have a couple of extremely long spectrograms you might want to use this
tts_bin_lengths = True              # bins the spectrogram lengths before sampling in data loader - speeds up training
tts_clip_grad_norm = 1.0            # clips the gradient norm to prevent explosion - set to None if not needed
tts_checkpoint_every = 2_000        # checkpoints the model every X steps
# TODO: tts_phoneme_prob = 0.0              # [0 <-> 1] probability for feeding model phonemes vrs graphemes


# ------------------------------------------------------------------------------------------------------------------#

# reference encoder
ref_enc_filters = [32, 32, 64, 64, 128, 128]
reference_depth = 128

# Global style token
num_gst = 10
num_heads = 4  # Head number for multi-head attention
style_embed_depth = 256
style_att_dim = 128
