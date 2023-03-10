import os
import torch

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3

####################################################################
######################## general variables #########################
####################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 20_000
MAX_SENT_LEN = 28 

####################################################################
######################## general autoencoder variables #############
####################################################################

ae_batch_size = 64
pretrained_embedding = True
bidirectional = True
attn_bool = True
num_attn_heads = 5
encoder_dim = 100
decoder_dim = 600
latent_dim = 100
dropout_prob = 0.5
layer_norm = True
ae_learning_rate = 1e-3
ae_betas = [0.9, 0.999]
word_embedding = 300

####################################################################
######################## cnn autoencoder variables #################
####################################################################

max_pool_kernel = 2
kernel_sizes = [3, 8]
out_channels = 10

####################################################################
######################## gan variables #############################
####################################################################

gan_batch_size = 64
n_layers = 20 
block_dim = 100
g_learning_rate = 1e-4
c_learning_rate = 5e-4
shared_learning_rate = 1e-4
