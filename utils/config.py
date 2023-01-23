import os
import torch

# general variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 20_000
batch_size = 100 # too high batch dimension may overload RAM

# vae variables
encoder_dim = 100
decoder_dim = 600
dropout_prob = 0.2
vae_learning_rate = 0.001
word_embedding = 200

# gan variables
mlp_hidden = 100