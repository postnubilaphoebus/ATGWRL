import numpy as np
import torch
from torch import nn
import utils.config as config
import time
import math
import numpy as np

MAX_SENT_LEN = 28

# Notation:
# B = Batch size
# S = Sequence length
# H = Hidden dimension
# E = Embedding dimension
# V = Vocab size
# L = Latent dimension

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.vocab_size = config.vocab_size
        self.encoder_dim = config.encoder_dim
        self.decoder_dim = config.decoder_dim
        self.embedding_dimension = config.word_embedding
        self.dropout = config.dropout_prob
        self.batch_size = config.batch_size
        self.latent_dim = config.latent_dim
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dimension)
        self.encoder_rnn = torch.nn.GRU(self.embedding_dimension, self.encoder_dim, batch_first=True, bidirectional = True)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.fast_decoder = torch.nn.GRUCell(self.decoder_dim, self.decoder_dim)
        self.code_layer = torch.nn.Linear(self.encoder_dim, self.latent_dim)
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)
        self.decoder_softmax = torch.nn.Softmax(dim=-1)
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.re_embed = torch.nn.Linear(self.decoder_dim, self.embedding_dimension)
        self.fc_1 = torch.nn.Linear(self.encoder_dim * 2, self.latent_dim) # because bidirectional
        self.fc_2 = torch.nn.Linear(self.latent_dim, self.decoder_dim)
        
    def init_weights(self):
        if isinstance(self, nn.Embedding):
            sqrt3 = math.sqrt(3)
            torch.nn.init.normal_(self.weight.data, std=sqrt3)
        elif isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)
        else:
            pass
        
    def encoder(self, x, x_lens, skip_embed = False):
        if not skip_embed:
            # [B, S] -> [B, S, E]
            embedded = self.embedding_layer(x)
        else:
            embedded = x

        # packing for efficiency and masking
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted = False)
        # [B, S, E] -> [B, S, H]
        output, _ = self.encoder_rnn(packed) 

        # unpack to extract last non-padded element
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length = MAX_SENT_LEN)
        
        # get the last (time-wise) hidden state of the encoder
        # [B, S, H] -> [B, H]
        context = []
        for sequence, unpadded_len in zip(output, x_lens):
            context.append(sequence[unpadded_len-1, :])
        context = torch.stack((context))
        
        # [B, H] -> [B, L]
        z = self.fc_1(context)
        
        # dropout
        z = self.dropout_layer(z)
        
        return z, embedded
        
    def decoder(self, z, true_inp, tf_prob):# x, tf_prob
        
        # to decoder dim
        output = self.fc_2(z)

        unmodded = output
        
        output = output.unsqueeze(0)
        
        # [B, H] -> [B, S, H]
        encoded = output.repeat(MAX_SENT_LEN, 1, 1)
        repeating = int(self.decoder_dim / self.embedding_dimension)
        
        true_inp = true_inp.repeat(1, 1, repeating)
        true_inp = torch.transpose(true_inp, 1, 0)
        
        hx = unmodded
        outs = []
        
        for i, inp in enumerate(encoded):
            rand_float = np.random.rand()
            if rand_float < tf_prob:
                # teacher forcing
                hx = self.fast_decoder(inp, true_inp[i])
            else:
                hx = self.fast_decoder(inp, hx)
                
            outs.append(hx)
            
        out = torch.stack((outs))
        
        return out
        
    
    def forward(self, x, x_lens, tf_prob = 0):
        
        # [B, S]-> [B, H]
        z, embedded = self.encoder(x, x_lens)
        
        # [B, L] -> [B, S, H]
        decoded = self.decoder(z, embedded, tf_prob)
        
        re_embedded = self.leaky_relu(self.re_embed(decoded))
        re_embedded = torch.transpose(re_embedded, 1, 0)
        
        # [B, S, H] -> [B, S, V]
        logits = self.hidden_to_vocab(decoded)

        return logits, z, re_embedded