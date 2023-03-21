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
attn_bool = False
num_attn_heads = 5
encoder_dim = 100
decoder_dim = 600
latent_dim = 100
dropout_prob = 0.5
layer_norm = True
ae_learning_rate = 5e-4
ae_betas = [0.9, 0.999]
word_embedding = 100

####################################################################
######################## cnn autoencoder variables #################
####################################################################

max_pool_kernel = 2
kernel_sizes = [3, 17]
out_channels = 100

####################################################################
######################## gan variables #############################
####################################################################

gan_batch_size = 128
n_layers = 10 
block_dim = 100
g_learning_rate = 1e-4
c_learning_rate = 3e-4
shared_learning_rate = 1e-4
