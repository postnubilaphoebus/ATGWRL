import numpy as np
import torch
from torch import nn
import utils.config as config
import time

MAX_SENT_LEN = 23 # 20 (+ 3 for punctuation)

# todo: sample from multivariate gaussian

'''
Once all symbols of the input are passed, 
the final hidden state is denoted as the summary c of the input. 
The decoder is an RNN that is used used to generate an output sequence y given a summary c. 
'''

# sth like this:
'''
from  torch.distributions import multivariate_normal
dist = multivariate_normal.MultivariateNormal(loc=torch.zeros(5), covariance_matrix=torch.eye(5))
probability = torch.exp(dist.log_prob(torch.randn(5)))
'''

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
        #self.decoder = torch.nn.LSTM(self.encoder_dim, self.decoder_dim, batch_first=True)
        self.layer_normalisation = torch.nn.LayerNorm(self.embedding_dimension)
        self.decoder = torch.nn.LSTMCell(self.encoder_dim, self.decoder_dim)
        self.linear_decoder = torch.nn.Linear(self.decoder_dim, self.vocab_size)
        self.decoder_softmax = torch.nn.Softmax(dim=-1)
        self.re_embedding_layer = torch.nn.Linear(self.decoder_dim, self.embedding_dimension)

    def reparameterize(self, z_mean, z_log_var):
        # z_mean.shape == z_log_var.shape == [batch, seqlen, encod_dim]
        eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(self.device)
        z = z_mean + eps * torch.exp(z_log_var/2.) 
        return z

    def init_hidden(self, x):
        hidden = torch.zeros(x.size(0), self.decoder_dim).to(self.device) 
        cell = torch.zeros(x.size(0), self.decoder_dim).to(self.device)
        return (hidden,cell)

    def autoregressive_decoding(self, encoded, train_mode):

        hn, cn = self.init_hidden(encoded)
        decoded_logits = []
        decoder_hidden = []
        if not train_mode:
            decoded_tokens = []
        else:
            decoded_tokens = None
            
        for idx in range(MAX_SENT_LEN):
            hn, cn = self.decoder(encoded, (hn, cn))
            decoder_hidden.append(self.re_embedding_layer(hn))
            logits = self.linear_decoder(hn)
            if not train_mode:
                tokens = torch.argmax(self.decoder_softmax(logits), dim = -1)
                decoded_tokens.append(tokens)
            decoded_logits.append(logits)
            
        if not train_mode:
            decoded_tokens = torch.stack((decoded_tokens))
        decoded_logits = torch.stack((decoded_logits))
        decoder_hidden = torch.stack((decoder_hidden))

        return decoded_tokens, decoded_logits, decoder_hidden

    
    def forward(self, x, x_lens, train_mode = False):

        # x.shape = [batch_size, seq_len]

        t0 = time.time()

        embedded = self.embedding_layer(x)
        embedded = self.dropout_layer(embedded)
        embedded = self.layer_normalisation(embedded)

        t1 = time.time()

        # pack sequence without 0s to speed up LSTM
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted = False)
        output, _ = self.encoder(embedded)

        # unpack sequence for further processing
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length = max(x_lens))
        
        context = []
        #batch_output is post_padded
        for sequence, unpadded_len in zip(output, x_lens):
            context.append(sequence[unpadded_len-1, :])
        context = torch.stack((context))

        # dropout
        #output = self.dropout_layer(output)

        # get the last hidden state (of the sequence) of the encoder
        # output.shape = [batch, seqlen, encoder_dim]
        #output = output[:, -1, :]
        output = context

        t2 = time.time()

        # code layer
        z_mean, z_log_var = self.z_mean(output), self.z_log_var(output)

        t3 = time.time()
        
        encoded = self.reparameterize(z_mean, z_log_var)

        t4 = time.time()

        # autoregressive decoding
        decoded_tokens, decoded_logits, _ = self.autoregressive_decoding(encoded, train_mode)

        t5 = time.time()

        decoding_time = t5 - t4
        reparam_time = t4 - t3
        code_layer_time = t3 - t2
        encoder_plus_dropout_time = t2 - t1
        embedding_time = t1 - t0

        time_list = [decoding_time, reparam_time, code_layer_time, encoder_plus_dropout_time, embedding_time]

        return encoded, z_mean, z_log_var, decoded_tokens, decoded_logits, time_list