import os
import torch

# general variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 20_000
batch_size = 16 # too high batch dimension may overload RAM

# vae variables
encoder_dim = 100
decoder_dim = 600
dropout_prob = 0.5
vae_learning_rate = 5e-4
word_embedding = 200

# gan variables
mlp_hidden = 100