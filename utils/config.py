import os
import torch

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3

# general variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 20_000
batch_size = 64 # too high batch dimension may overload RAM
MAX_SENT_LEN = 28 

# ae variables
encoder_dim = 100
decoder_dim = 600
latent_dim = 100
dropout_prob = 0.5
ae_learning_rate = 1e-3#5e-4
word_embedding = 200
ae_batch_size = 64

# gan variables
n_layers = 10 
block_dim = 100
g_learning_rate = 1e-3
c_learning_rate = 2e-4
gan_learning_rate = 1e-4