import os
import torch

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3

# general variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 30_000
batch_size = 256 # too high batch dimension may overload RAM
MAX_SENT_LEN = 28 

# ae variables
encoder_dim = 100
decoder_dim = 600
latent_dim = 100
dropout_prob = 0.5
ae_learning_rate = 5e-4
word_embedding = 200

# gan variables
n_layers = 20 
block_dim = 100
gan_learning_rate = 1e-4