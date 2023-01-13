import numpy as np
import torch
from torch import nn
import utils.config as config

MAX_SENT_LEN = 30

class VariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.vocab_size = config.vocab_size
        self.encoder_dim = config.encoder_dim
        self.decoder_dim = config.decoder_dim
        self.embedding_dimension = config.word_embedding
        self.dropout = config.dropout_prob
        self.batch_size = config.batch_size
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dimension)
        self.encoder = torch.nn.LSTM(self.embedding_dimension, self.encoder_dim)
        self.z_mean = torch.nn.Linear(self.encoder_dim, self.encoder_dim)
        self.z_log_var = torch.nn.Linear(self.encoder_dim, self.encoder_dim)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.decoder = torch.nn.LSTM(self.encoder_dim, self.decoder_dim)
        self.linear_decoder = torch.nn.Linear(self.decoder_dim, self.vocab_size)
        self.decoder_softmax = torch.nn.Softmax(dim=-1)

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn(z_mean.size(0), z_mean.size(1), z_mean.size(2)).to(self.device)
        z = z_mean + eps * torch.exp(z_log_var/2.) 
        return z

    def init_hidden(self, x):
        hidden = torch.zeros(1, x.size(1), self.decoder_dim).to(self.device) 
        cell = torch.zeros(1, x.size(1), self.decoder_dim).to(self.device)
        return (hidden,cell)

    def autoregressive_decoding(self, encoded):

        hn, cn = self.init_hidden(encoded)
        decoded_tokens = []
        decoded_logits = []
        idx = 0
        while idx < MAX_SENT_LEN:
            output, hiddens = self.decoder(encoded, (hn, cn))
            h_n, c_n = hiddens
            logits = self.linear_decoder(output)
            tokens = torch.argmax(self.decoder_softmax(logits), dim = -1)
            decoded_tokens.append(tokens)
            decoded_logits.append(logits)
            idx += 1

        decoded_tokens = torch.stack((decoded_tokens))
        decoded_logits = torch.stack((decoded_logits))

        return decoded_tokens, decoded_logits

    
    def forward(self, x, x_lens):
        embedded = self.embedding_layer(x)

        # pack sequence without 0s to speed up LSTM
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted = False)
        output, _ = self.encoder(embedded)

        # unpack sequence for further processing
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length = max(x_lens))

        # dropout
        output = self.dropout_layer(output)

        # code layer
        z_mean, z_log_var = self.z_mean(output), self.z_log_var(output)
        encoded = self.reparameterize(z_mean, z_log_var)

        # autoregressive decoding
        decoded_tokens, decoded_logits = self.autoregressive_decoding(encoded)
        
        return encoded, z_mean, z_log_var, decoded_tokens, decoded_logits