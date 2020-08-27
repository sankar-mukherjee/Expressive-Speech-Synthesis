import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init


######################################


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_units = num_units

        self.W_query = nn.Conv1d(in_channels=query_dim, out_channels=num_units, kernel_size=1)
        self.W_key = nn.Conv1d(in_channels=key_dim, out_channels=num_units, kernel_size=1)

    def forward(self, query, key):
        value = key

        query = query.transpose(2, 1)
        key = key.transpose(2, 1)
        query = self.W_query(query)
        key = self.W_key(key)
        query = query.transpose(2, 1)
        key = key.transpose(2, 1)

        # print(querys.shape, keys.shape, values.shape)
        split_size = self.num_units // self.num_heads
        query = torch.stack(torch.split(query, split_size, dim=2), dim=1)
        key = torch.stack(torch.split(key, split_size, dim=2), dim=1)
        value = value.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # print('----------------------------')
        # print(query.shape, key.shape, value.shape)
        context = self._mlp_attention(query, key, value)

        style_embeddings = context.view(context.size(0), context.size(2), -1)
        # style_embeddings = context.reshape(context.shape[0], 1, context.shape[1] * context.shape[-1])

        # print(context.shape, style_embeddings.shape)
        return style_embeddings

    def _mlp_attention(self, qs, ks, vs):
        num_units = qs.shape[-1]
        # Scalar used in weight normalization
        g = math.sqrt((1. / num_units))
        # Bias added prior to the nonlinearity
        b = torch.zeros(num_units).cuda()
        v = torch.randn(num_units).cuda()

        normed_v = g * v * torch.rsqrt(torch.sum(torch.square(v)))
        # Single layer multilayer perceptron.
        add = torch.sum(normed_v * F.tanh(ks + qs + b), [-1], keepdim=True)
        # Compute attention weights.
        weights = F.softmax(add.transpose(3, 2))
        # Compute attention context.
        context = torch.matmul(weights, vs)
        # print(weights.shape, add.shape, context.shape)
        return context


class STL(nn.Module):
    def __init__(self, num_gst, style_embed_depth, num_heads, num_units):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(num_gst, style_embed_depth // num_heads))  # 10 x 64

        init.normal_(self.embed, mean=0, std=0.5)
        # # d_q depends on refnet_outputs or end layer size of ref_enc_filters
        d_q = 128
        d_k = style_embed_depth // num_heads
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=num_units, num_heads=num_heads)

    def forward(self, x):
        N = x.size(0)
        query = x.unsqueeze(1)  # batch x 1 x 128
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # batch x 10 x 64
        # print(query.shape, keys.shape)

        style_embeddings = self.attention(query, keys)  # batch x 1 x 256
        return style_embeddings


class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2)):
        super().__init__()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x


class ReferenceEncoder(nn.Module):
    def __init__(self, ref_enc_filters, n_mels, reference_depth):
        super().__init__()
        filters = [1] + ref_enc_filters
        self.layers = nn.ModuleList()
        for i in range(len(filters) - 1):
            self.layers.append(BatchNormConv2d(in_channels=filters[i], out_channels=filters[i + 1]))

        out_channels = self.calculate_channels(n_mels, 3, 2, 1, len(ref_enc_filters))
        self.gru = nn.GRU(input_size=filters[-1] * out_channels,
                          hidden_size=reference_depth,
                          batch_first=True)
        self.linear = nn.Linear(in_features=128, out_features=reference_depth)

    def forward(self, m):
        x = m.unsqueeze(1)
        for f in self.layers: x = f(x)
        x = x.view(m.size(0), x.size(-1), -1)
        encoder_outputs, encoder_state = self.gru(x)
        # print(encoder_outputs.shape, encoder_state.shape)
        x = self.linear(encoder_outputs[:, -1, :])
        # print(x.shape, encoder_outputs[:, -1, :].shape)
        x = F.tanh(x)
        return x

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


######################################


class Encoder(nn.Module):
    def __init__(self, embed_dims, num_chars, cbhg_channels, K, num_highways, dropout,
                 ref_enc_filters, n_mels, reference_depth,
                 num_gst, num_heads, style_embed_depth, style_att_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels,
                         proj_channels=[cbhg_channels, cbhg_channels],
                         num_highways=num_highways)
        # ref encoder
        self.ref_enc = ReferenceEncoder(ref_enc_filters=ref_enc_filters, n_mels=n_mels, reference_depth=reference_depth)
        # Style attention
        self.stl = STL(num_gst=num_gst, style_embed_depth=style_embed_depth, num_heads=num_heads,
                       num_units=style_att_dim)

    def forward(self, x, m):
        ################################ after_embedding #########################
        x = self.embedding(x)
        x = self.pre_net(x)
        x.transpose_(1, 2)
        encoder_outputs = self.cbhg(x)  # batch x -- x 256

        # # reference encoder
        ref_encoder_outputs = self.ref_enc(m)  # batch x 128
        # print(encoder_outputs.shape, ref_encoder_outputs.shape)
        # GST Style attention
        style_embeddings = self.stl(ref_encoder_outputs)  # 8 x 1 x 256 for custom multihead
        # Add style embedding to every text encoder state
        style_embeddings = style_embeddings.expand(-1, encoder_outputs.shape[1], -1)
        encoder_outputs = torch.cat((encoder_outputs, style_embeddings), -1)
        # print(encoder_outputs.shape)
        # print('----------------')

        ################################## before_embedding ################################
        # x = self.embedding(x)
        # # reference encoder
        # ref_encoder_outputs = self.ref_enc(m)
        # # GST Style attention
        # style_embeddings = self.stl(ref_encoder_outputs)
        # # Add style embedding to every text encoder state
        # style_embeddings = style_embeddings.expand(-1, x.shape[1], -1)
        # x = torch.cat((x, style_embeddings), -1)
        #
        # x = self.pre_net(x)
        # x.transpose_(1, 2)
        # encoder_outputs = self.cbhg(x)
        return encoder_outputs


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y


class BatchNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class CBHG(nn.Module):
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()

        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

        # Fix the highway input if necessary
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False

        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)

    def forward(self, x):

        # Save these for later
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.conv1d_bank:
            c = conv(x)  # Convolution
            conv_bank.append(c[:, :, :seq_len])

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)[:, :, :seq_len]

        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
        for h in self.highways: x = h(x)

        # And then the RNN
        x, _ = self.rnn(x)
        return x


class PreNet(nn.Module):
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x


class Attention(nn.Module):
    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        self.v = nn.Linear(attn_dims, 1, bias=False)

    def forward(self, encoder_seq_proj, query, t):
        # print(encoder_seq_proj.shape)
        # Transform the query vector
        query_proj = self.W(query).unsqueeze(1)

        # Compute the scores
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))
        scores = F.softmax(u, dim=1)

        return scores.transpose(1, 2)


class LSA(nn.Module):
    def __init__(self, attn_dim, kernel_size=31, filters=32):
        super().__init__()
        self.conv = nn.Conv1d(2, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=False)
        self.L = nn.Linear(filters, attn_dim, bias=True)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None

    def init_attention(self, encoder_seq_proj):
        b, t, c = encoder_seq_proj.size()
        self.cumulative = torch.zeros(b, t).cuda()
        self.attention = torch.zeros(b, t).cuda()

    def forward(self, encoder_seq_proj, query, t):
        if t == 0: self.init_attention(encoder_seq_proj)

        processed_query = self.W(query).unsqueeze(1)

        location = torch.cat([self.cumulative.unsqueeze(1), self.attention.unsqueeze(1)], dim=1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))

        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
        u = u.squeeze(-1)

        # Smooth Attention
        scores = torch.sigmoid(u) / torch.sigmoid(u).sum(dim=1, keepdim=True)
        # scores = F.softmax(u, dim=1)
        self.attention = scores
        self.cumulative += self.attention

        return scores.unsqueeze(-1).transpose(1, 2)


class Decoder(nn.Module):
    def __init__(self, n_mels, decoder_dims, lstm_dims):
        super().__init__()
        self.max_r = 20
        self.r = None
        self.generating = False
        self.n_mels = n_mels
        self.prenet = PreNet(n_mels, fc2_dims=256)
        self.attn_net = LSA(decoder_dims)
        self.attn_rnn = nn.GRUCell(decoder_dims + decoder_dims // 2, decoder_dims)
        self.rnn_input = nn.Linear(2 * decoder_dims, lstm_dims)
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)

    def zoneout(self, prev, current, p=0.1):
        mask = torch.zeros(prev.size()).bernoulli_(p).cuda()
        return prev * mask + current * (1 - mask)

    def forward(self, encoder_seq, encoder_seq_proj, prenet_in,
                hidden_states, cell_states, context_vec, t):

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in)

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)
        #print(context_vec.shape, attn_rnn_in.shape, attn_hidden.shape)

        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)

        # Dot product to create the context vector
        context_vec = scores @ encoder_seq
        context_vec = context_vec.squeeze(1)
        # print(context_vec.shape, attn_hidden.shape)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden], dim=1)
        #print(x.shape, context_vec.shape, attn_hidden.shape)

        x = self.rnn_input(x)

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if not self.generating:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if not self.generating:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, hidden_states, cell_states, context_vec


class Tacotron(nn.Module):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout,
                 ref_enc_filters, reference_depth,
                 num_gst, num_heads, style_embed_depth, style_att_dim):
        super().__init__()
        self.n_mels = n_mels
        self.lstm_dims = lstm_dims
        self.decoder_dims = decoder_dims
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims,
                               encoder_K, num_highways, dropout,
                               ref_enc_filters, n_mels, reference_depth,
                               num_gst, num_heads, style_embed_depth, style_att_dim)

        # for custom multihead
        self.encoder_proj = nn.Linear(decoder_dims, decoder_dims, bias=False)
        self.decoder = Decoder(n_mels, decoder_dims, lstm_dims)
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims, [256, 80], num_highways)
        self.post_proj = nn.Linear(postnet_dims * 2, fft_bins, bias=False)

        self.init_model()
        self.num_params()

        # Unfortunately I have to put these settings into params in order to save
        # if anyone knows a better way of doing this please open an issue in the repo
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        self.r = nn.Parameter(torch.tensor(0).long(), requires_grad=False)

    def set_r(self, r):
        self.r.data = torch.tensor(r)
        self.decoder.r = r

    def get_r(self):
        return self.r.item()

    def forward(self, x, m, generate_gta=False):

        self.step += 1

        if generate_gta:
            self.encoder.eval()
            self.postnet.eval()
            self.decoder.generating = True
        else:
            self.encoder.train()
            self.postnet.train()
            self.decoder.generating = False

        batch_size, _, steps = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels).cuda()

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims).cuda()

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x, m)
        #print(encoder_seq.shape)
        encoder_seq_proj = self.encoder_proj(encoder_seq)
        # print(encoder_seq_proj.shape)

        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []

        # Run the decoder loop
        for t in range(0, steps, self.r):
            prenet_in = m[:, :, t - 1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores

    def generate(self, x, steps=2000):

        self.encoder.eval()
        self.postnet.eval()
        self.decoder.generating = True

        batch_size = 1
        x = torch.LongTensor(x).unsqueeze(0).cuda()

        # Need to initialise all hidden states and pack into tuple for tidyness
        attn_hidden = torch.zeros(batch_size, self.decoder_dims).cuda()
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims).cuda()
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Need to initialise all lstm cell states and pack into tuple for tidyness
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims).cuda()
        cell_states = (rnn1_cell, rnn2_cell)

        # Need a <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels).cuda()

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims).cuda()

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x, context_vec)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []

        # Run the decoder loop
        for t in range(0, steps, self.r):
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            # Stop the loop if silent frames present
            if (mel_frames < -3.8).all() and t > 10: break

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)

        linear = linear.transpose(1, 2)[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()[0]

        self.encoder.train()
        self.postnet.train()
        self.decoder.generating = False

        return mel_outputs, linear, attn_scores

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)

    def checkpoint(self, path):
        k_steps = self.get_step() // 1000
        self.save(f'{path}/checkpoint_{k_steps}k_steps.pyt')

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def restore(self, path):
        if not os.path.exists(path):
            print('\nNew Tacotron Training Session...\n')
            self.save(path)
        else:
            print(f'\nLoading Weights: "{path}"\n')
            self.load(path)
            self.decoder.r = self.r.item()

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
