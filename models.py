import numpy as np
import torch
from torch import nn
import math
import numpy as np
import sys
import torch.nn.utils.spectral_norm as spectral_norm

MAX_SENT_LEN = 28

# Notation:
# B = Batch size
# S = Sequence length
# H = Hidden dimension
# E = Embedding dimension
# V = Vocab size
# L = Latent dimension

##########################################################################################################
###                                                                                                    ###
########################################## General Model functions #######################################                                      
###                                                                                                    ###
##########################################################################################################

def create_emb_layer(weights_matrix, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer

##########################################################################################################
###                                                                                                    ###
########################################## Autoencoder models ############################################                                         
###                                                                                                    ###
##########################################################################################################

class DefaultDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.decoder_dim = config.decoder_dim
        self.latent_dim = config.latent_dim
        self.layer_norm = config.layer_norm
        self.hidden_dim = config.decoder_dim
        self.embedding_dimension = config.word_embedding
        self.fc = torch.nn.Linear(self.latent_dim, self.decoder_dim)
        self.layer_normalisation = torch.nn.LayerNorm(self.decoder_dim)
        self.decoder_rnn = torch.nn.GRU(self.decoder_dim, self.decoder_dim, batch_first=True)

    def forward(self, z, true_inp, tf_prob):
        # [B, L] -> [B, H]
        output = self.fc(z)

        # [B, H] -> [B, S, H]
        output, _ = self.decoder_rnn(output.unsqueeze(1).repeat(1, MAX_SENT_LEN, 1), output.unsqueeze(0))

        if self.layer_norm == True:
            output = self.layer_normalisation(output)

        # [B, S, H] -> [S, B, H]
        output = torch.transpose(output, 1, 0)
        
        return output
    
class ExperimentalDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder_dim = config.decoder_dim
        self.latent_dim = config.latent_dim
        self.vocab_size = config.vocab_size
        self.num_mlps = 9
        self.attn_heads = 3
        self.attn_softmax = nn.Softmax(dim = -1)
        self.mlp_layers = nn.ModuleList([nn.Linear(self.latent_dim, self.decoder_dim) \
                            for _ in range(self.num_mlps)])
        self.activation_function = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_layer = nn.Conv1d(in_channels = self.decoder_dim, out_channels = self.decoder_dim, 
                                    kernel_size = 3, groups = 1)
        self.attn_layers = nn.ModuleList([nn.Linear(self.decoder_dim, self.decoder_dim) \
                                          for _ in range(self.attn_heads)])
        self.attn_out = nn.Linear(self.decoder_dim * self.attn_heads, self.decoder_dim)
        self.upsample_layers = nn.ModuleList([nn.Linear(self.decoder_dim * 2, self.decoder_dim) \
                                              for _ in range(4)])
        self.fc = nn.Linear(self.decoder_dim, self.decoder_dim)
        
    def attention(self, x):
        layer_out = []
        for layer in self.attn_layers:
            x_out = layer(x)
            u_t = torch.tanh(x_out)
            attn_weights = self.attn_softmax(u_t)
            attended = attn_weights * x
            attended = torch.sum(attended, dim = 1)
            layer_out.append(attended)
        layer_out = torch.cat((layer_out), dim = -1)
        attended = self.attn_out(layer_out)
        attended = attended.unsqueeze(1)
        attended = attended.repeat(1, attn_weights.size(1), 1)
        return attended

    def forward(self, z, true_inp, tf_prob):
        output = []
        for layer in self.mlp_layers:
            out = self.activation_function(layer(z))
            output.append(out)
        output = torch.stack((output))
        output = torch.transpose(output, -1, 0)
        output = torch.transpose(output, 1, 0)
        output = self.conv_layer(output)
        output = torch.transpose(output, 2, 1)
        attended_out = self.attention(output)
        output = torch.cat((output, attended_out), dim = -1)
        upsampled = []
        for layer in self.upsample_layers:
            out = self.activation_function(layer(output))
            upsampled.append(out)
        upsampled = torch.cat((upsampled), dim = 1)
        output = self.fc(upsampled)
        output = torch.transpose(output, 1, 0)

        return output
    
class CNN_Encoder(nn.Module):

    def __init__(self, config, weights_matrix = None):
        super().__init__()
        assert type(config.kernel_sizes) is list, "kernel sizes must be list"
        assert len(config.kernel_sizes) == 2, "two kernel sizes required"
        self.max_pool_kernel = config.max_pool_kernel
        self.cnn_embed = config.word_embedding
        self.kernel_sizes = config.kernel_sizes 
        self.out_channels = config.out_channels
        self.latent_dim = config.latent_dim
        self.dropout_prob = config.dropout_prob
        self.use_dropout = config.use_dropout
        self.vocab_size = config.vocab_size
        self.embedding_layer = nn.Embedding(self.vocab_size, self.cnn_embed) if weights_matrix == None \
                               else create_emb_layer(weights_matrix, non_trainable=True)
        self.first_convolution_a = nn.Conv1d(in_channels = self.cnn_embed, out_channels = self.out_channels, 
                                             kernel_size = self.kernel_sizes[0], groups = 1)
        self.first_convolution_b = nn.Conv1d(in_channels = self.cnn_embed, out_channels = self.out_channels, 
                                             kernel_size = self.kernel_sizes[1], groups = 1)
        self.groupNorm_a = torch.nn.GroupNorm(10, self.out_channels)
        self.groupNorm_b = torch.nn.GroupNorm(10, self.out_channels)
        self.l_out_a = (MAX_SENT_LEN - (self.kernel_sizes[0] - 1) - (self.max_pool_kernel - 1) - 1) / self.max_pool_kernel + 1
        self.l_out_b = (MAX_SENT_LEN - (self.kernel_sizes[1] - 1) - (self.max_pool_kernel - 1) - 1) / self.max_pool_kernel + 1
        self.max_pool = nn.MaxPool1d(kernel_size = self.max_pool_kernel, stride = self.max_pool_kernel)
        self.to_latent = torch.nn.Linear(round((self.l_out_a + self.l_out_b) * self.out_channels), self.latent_dim)
        self.second_dropout = nn.Dropout(p = self.dropout_prob, inplace=False)

    def forward(self, x, mixed_up_batch, use_mixup):

        if use_mixup == False:
            embedded = self.embedding_layer(x)
        else:
            embedded = mixed_up_batch

        embedded_t = torch.transpose(embedded, 2, 1)
        c_1_a = self.first_convolution_a(embedded_t)
        c_1_a = self.max_pool(c_1_a)
        c_1_a = self.groupNorm_a(c_1_a)
        c_1_a = torch.flatten(c_1_a, start_dim = 1)
        c_1_b = self.first_convolution_b(embedded_t)
        c_1_b = self.max_pool(c_1_b)
        c_1_b = self.groupNorm_b(c_1_b)
        c_1_b = torch.flatten(c_1_b, start_dim = 1)
        c_1_a = torch.transpose(c_1_a, 1, 0)
        c_1_b = torch.transpose(c_1_b, 1, 0)
        c_1 = torch.cat((c_1_a, c_1_b))
        c_1 = torch.transpose(c_1, 1, 0)

        z = self.to_latent(c_1)
        if self.use_dropout == True:
            z = self.second_dropout(z)

        return z, embedded
    
class DefaultEncoder(nn.Module):
    def __init__(self, config, weights_matrix = None):
        super().__init__()
        self.device = config.device
        self.vocab_size = config.vocab_size
        self.layer_norm = config.layer_norm
        self.embedding_dimension = config.word_embedding
        self.encoder_dim = config.encoder_dim
        self.latent_dim = config.latent_dim
        self.dropout = config.dropout_prob
        self.use_dropout = config.use_dropout
        self.hidden_dim = config.encoder_dim
        self.embedding_layer = nn.Embedding(config.vocab_size, self.embedding_dimension) if weights_matrix == None \
                               else create_emb_layer(weights_matrix, non_trainable=True)
        self.bidirectional = config.bidirectional
        self.attn_multiplier = 2 if self.bidirectional else 1
        self.encoder_rnn = torch.nn.GRU(self.embedding_dimension, self.encoder_dim, batch_first=True, bidirectional = self.bidirectional)
        self.layer_normalisation = torch.nn.LayerNorm(self.encoder_dim * self.attn_multiplier)
        self.attn_softmax = nn.Softmax(dim = -1)
        self.attn_bool = config.attn_bool
        self.attn_heads = config.num_attn_heads
        self.attn_layers = nn.ModuleList([nn.Linear(self.attn_multiplier * self.encoder_dim, self.attn_multiplier * self.encoder_dim) \
                            for _ in range(self.attn_heads)])
        self.attn_out = nn.Linear(self.attn_heads * self.attn_multiplier * self.encoder_dim , self.attn_multiplier * self.encoder_dim)
        self.layer_multiplier = 4 if self.attn_bool else self.attn_multiplier # because bidirectional & attention
        self.fc_1 = torch.nn.Linear(self.encoder_dim * self.layer_multiplier, self.latent_dim) 
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def mask_padding(self, x_lens, hidden_dim):
        max_seq_len = MAX_SENT_LEN
        mask = []
        for seq_len in x_lens:
            a = [0 for _ in range(seq_len)]
            if max_seq_len - seq_len > 0:
                b = [1 for _ in range(max_seq_len - seq_len)]
                a = a + b 
            mask.append(a)
        mask = torch.BoolTensor(mask).to(self.device)
        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1, 1, hidden_dim)
        return mask

    def attention(self, x, mask):
        # attention from Yang, 2016
        # https://aclanthology.org/N16-1174.pdf
        # using multi-head
        layer_out = []
        for layer in self.attn_layers:
            x_out = layer(x)
            u_t = torch.tanh(x_out)
            attn_weights = self.attn_softmax(u_t)
            attn_weights = attn_weights.masked_fill(mask, -1e12)
            attended = attn_weights * x
            attended = torch.sum(attended, dim = 1)
            layer_out.append(attended)
        layer_out = torch.cat((layer_out), dim = -1)
        attended = self.attn_out(layer_out)
        attended = attended.unsqueeze(1)
        attended = attended.repeat(1, attn_weights.size(1), 1)
        return attended

    def forward(self, x, x_lens, mixed_up_batch, use_mixup):# mixed_up_batch = None, use_mixup = False

        # [B, S] -> [B, S, E]
        if use_mixup == False:
            embedded = self.embedding_layer(x)
        else:
            embedded = mixed_up_batch
        # packing for efficiency and masking
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted = False)
        # [B, S, E] -> [B, S, H]
        output, _ = self.encoder_rnn(packed) 

        # unpack to extract last non-padded element
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length = MAX_SENT_LEN)

        if self.layer_norm == True:
            output = self.layer_normalisation(output)

        if self.attn_bool:
            mask = self.mask_padding(x_lens, output.size(-1))
            attn_vector = self.attention(output, mask)
            output = torch.cat((output, attn_vector), dim = -1)
 
        # get the last (time-wise) hidden state of the encoder
        # [B, S, H] -> [B, H]
        context = []
        for sequence, unpadded_len in zip(output, x_lens):
            context.append(sequence[unpadded_len-1, :])
        context = torch.stack((context))
        
        # [B, H] -> [B, L]
        z = self.fc_1(context)

        if self.use_dropout == True:
            # dropout
            z = self.dropout_layer(z)
        
        return z, embedded

class AutoEncoder(nn.Module):
    def __init__(self, config, weights_matrix = None):
        super().__init__()
        self.name = "default_autoencoder"
        self.device = config.device
        self.decoder_dim = config.decoder_dim
        self.vocab_size = config.vocab_size
        self.encoder = DefaultEncoder(config, weights_matrix)
        self.decoder = DefaultDecoder(config)
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)
        
    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)       
        else:
            pass
        
    def forward(self, x, x_lens, tf_prob = 0, mixed_up_batch = None, use_mixup = False):
        
        # [B, S] -> [B, H]
        z, embedded = self.encoder(x, x_lens, mixed_up_batch, use_mixup)
        
        # [B, L] -> [S, B, H]
        decoded = self.decoder(z, embedded, tf_prob)
        
        # [S, B, H] -> [S, B, V]
        logits = self.hidden_to_vocab(decoded)

        return logits
    
class CNNAutoEncoder(nn.Module):
    def __init__(self, config, weights_matrix = None):
        super().__init__()
        self.name = "cnn_autoencoder"
        self.device = config.device
        self.decoder_dim = config.decoder_dim
        self.vocab_size = config.vocab_size
        self.encoder = CNN_Encoder(config, weights_matrix)
        self.decoder = DefaultDecoder(config)
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)
        
    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)
        elif isinstance(self, nn.Conv1d):
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            pass
        
    def forward(self, x, x_lens, tf_prob = 0, mixed_up_batch = None, use_mixup = False):
        
        # [B, S] -> [B, H]
        z, embedded = self.encoder(x, mixed_up_batch, use_mixup)
        
        # [B, L] -> [S, B, H]
        decoded = self.decoder(z, embedded, tf_prob)
        
        # [S, B, H] -> [S, B, V]
        logits = self.hidden_to_vocab(decoded)

        return logits
    
class ExperimentalAutoencoder(nn.Module):
    def __init__(self, config, weights_matrix = None):
        super().__init__()
        self.name = "experimental_autoencoder"
        self.device = config.device
        self.decoder_dim = config.decoder_dim
        self.vocab_size = config.vocab_size
        self.encoder = CNN_Encoder(config, weights_matrix)
        self.decoder = ExperimentalDecoder(config)
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)
        elif isinstance(self, nn.Conv1d):
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            pass

    def forward(self, x, x_lens, tf_prob = 0):
        
        # [B, S] -> [B, H]
        z, embedded = self.encoder(x)
        
        # [B, L] -> [S, B, H]
        decoded = self.decoder(z, embedded, tf_prob)
        
        # [S, B, H] -> [S, B, V]
        logits = self.hidden_to_vocab(decoded)

        return logits
    
class VariationalAutoEncoder(nn.Module):
    def __init__(self, config, weights_matrix):
        super().__init__()
        self.name = "variational_autoencoder"
        self.device = config.device
        self.latent_dim = config.latent_dim
        self.vocab_size = config.vocab_size
        self.encoder_dim = config.encoder_dim
        self.decoder_dim = config.decoder_dim
        self.bidirectional = config.bidirectional
        self.embedding_dimension = config.word_embedding
        self.layer_norm = config.layer_norm
        self.dropout = config.dropout_prob
        self.use_dropout = config.use_dropout
        self.num_layers = 2
        self.layer_multiplier = 2 if self.bidirectional else 1
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dimension) if weights_matrix == None \
                               else create_emb_layer(weights_matrix, non_trainable=True)
        self.encoder_rnn = torch.nn.GRU(self.embedding_dimension, 
                                        self.encoder_dim, 
                                        num_layers = self.num_layers,
                                        batch_first=True, 
                                        bidirectional = self.bidirectional)
        self.layer_norm_encoder = torch.nn.LayerNorm(self.encoder_dim * self.layer_multiplier)
        self.to_latent = torch.nn.Linear(self.encoder_dim * self.layer_multiplier, self.latent_dim)
        self.z_mean = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.z_log_var = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.to_decoder = torch.nn.Linear(self.latent_dim, self.decoder_dim)
        self.decoder_rnn = torch.nn.GRU(self.decoder_dim, 
                                        self.decoder_dim, 
                                        num_layers = self.num_layers,
                                        batch_first=True)
        self.layer_norm_decoder = torch.nn.LayerNorm(self.decoder_dim)
        self.hidden_to_vocab = torch.nn.Linear(self.decoder_dim, self.vocab_size)

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)

    def encoder(self, x, x_lens, mixed_up_batch, use_mixup):

        # [B, S] -> [B, S, E]
        if use_mixup == False:
            embedded = self.embedding_layer(x)
        else:
            embedded = mixed_up_batch

        # packing for masking and speed
        embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted = False)
        # [B, S, E] -> [B, S, H]
        output, _ = self.encoder_rnn(embedded)

        # unpack sequence for further processing
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length = max(x_lens))

        if self.layer_norm == True:
            output = self.layer_norm_encoder(output)

        output = self.to_latent(output)

        if self.use_dropout == True:
            output = self.dropout_layer(output)

        return output

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(self.device)
        z = z_mean + eps * torch.exp(z_log_var/2.) 
        return z
    
    def holistic_regularisation(self, encoder_output):
        z_mean_list = []
        z_log_var_list = []
        for time_step in encoder_output:
            z_mean_list.append(self.z_mean(time_step))
            z_log_var_list.append(self.z_log_var(time_step))
        z_mean_list = torch.transpose(torch.stack((z_mean_list)), 1, 0)
        z_log_var_list = torch.transpose(torch.stack((z_log_var_list)), 1, 0)
        return z_mean_list, z_log_var_list
    
    def decoder(self, z):
        z = self.to_decoder(z)
        out, _ = self.decoder_rnn(z.unsqueeze(1).repeat(1,MAX_SENT_LEN,1), z.unsqueeze(0).repeat(self.num_layers, 1, 1))
        if self.layer_norm == True:
            out = self.layer_norm_decoder(out)
        logits = self.hidden_to_vocab(out)
        return logits
    
    def forward(self, x, x_lens, mixed_up_batch = None, use_mixup = False):

        # [B, S] -> [B, S, H]
        output = self.encoder(x, x_lens, mixed_up_batch, use_mixup)

        # [B, S, H] -> [S, B, H]
        output = torch.transpose(output, 1, 0)

        # [S, B, H] -> [B, S, H] (order to extract last time-step)
        z_mean_list, z_log_var_list = self.holistic_regularisation(output)

        # extract h_t-1
        z_mean_context = []
        z_log_var_context = []
        for z_mean_seq, z_log_var_seq, unpadded_len in zip(z_mean_list, z_log_var_list, x_lens):
            z_mean_context.append(z_mean_seq[unpadded_len-1, :])
            z_log_var_context.append(z_log_var_seq[unpadded_len-1, :])

        z_mean_context = torch.stack((z_mean_context))
        z_log_var_context = torch.stack((z_log_var_context))

        z = self.reparameterize(z_mean_context, z_log_var_context)

        # [B, S, H] -> [S, B, V] (order for reconstruction error)
        logits = torch.transpose(self.decoder(z), 1, 0)

        # [B, S, H] -> [S, B, H] (order for kl divergence)
        z_mean_list = torch.transpose(z_mean_list, 1, 0)
        z_log_var_list = torch.transpose(z_log_var_list, 1, 0)

        return z_mean_list, z_log_var_list, logits
    
##########################################################################################################
###                                                                                                    ###
########################################## GAN models ############################################     ###
###                                                                                                    ###
##########################################################################################################
    
class Block(nn.Module):
    
    def __init__(self, block_dim, activation_function = "relu", slope = 0.1, snm = True):
        super().__init__()
        if activation_function == "relu":
            if snm == False:
                self.net = nn.Sequential(
                    nn.Linear(block_dim, block_dim),
                    nn.ReLU(True),
                    nn.Linear(block_dim, block_dim),
                )
            else:
                self.net = nn.Sequential(
                    spectral_norm(nn.Linear(block_dim, block_dim)),
                    nn.ReLU(True),
                    spectral_norm(nn.Linear(block_dim, block_dim)),
                )
        elif activation_function == "leaky_relu":
            if snm == False:
                self.net = nn.Sequential(
                    nn.Linear(block_dim, block_dim),
                    nn.ReLU(True),
                    nn.Linear(block_dim, block_dim),
                )
            else:
                self.net = nn.Sequential(
                    spectral_norm(nn.Linear(block_dim, block_dim)),
                    nn.ReLU(True),
                    spectral_norm(nn.Linear(block_dim, block_dim)),
                )
        else:
            sys.exit("Please provide valid activation function.\
                      Choose among 'relu' and 'leaky_relu'. \
                     'leaky_relu' slope defaults to 0.1")
    
    def forward(self, x):
        return self.net(x) + x
    
class X_Skip_Block(nn.Module):
    
    def __init__(self, block_dim, activation_function = "relu", slope = 0.1):
        super().__init__()
        if activation_function == "relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.ReLU(True),
                nn.Linear(block_dim, block_dim),
            )
        elif activation_function == "leaky_relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.LeakyReLU(negative_slope = slope, inplace=True),
                nn.Linear(block_dim, block_dim),
                )
        else:
            sys.exit("Please provide valid activation function.\
                      Choose among 'relu' and 'leaky_relu'. \
                     'leaky_relu' slope defaults to 0.1")
    
    def forward(self, x):
        return self.net(x) + 3 * x
    
class LongerBlock(nn.Module):
    
    def __init__(self, block_dim, activation_function = "relu", slope = 0.1):
        super().__init__()
        if activation_function == "relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.ReLU(True),
                nn.Linear(block_dim, block_dim),
                nn.ReLU(True),
                nn.Linear(block_dim, block_dim),
            )
        elif activation_function == "leaky_relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.LeakyReLU(negative_slope = slope, inplace=True),
                nn.Linear(block_dim, block_dim),
                nn.LeakyReLU(negative_slope = slope, inplace=True),
                nn.Linear(block_dim, block_dim),
                )
        else:
            sys.exit("Please provide valid activation function.\
                      Choose among 'relu' and 'leaky_relu'. \
                     'leaky_relu' slope defaults to 0.1")
    
    def forward(self, x):
        return self.net(x) + x
    
class RecursiveLayerNormBlock(nn.Module):
    
    def __init__(self, block_dim, activation_function = "relu", slope = 0.1):
        super().__init__()
        if activation_function == "relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.ReLU(True),
                nn.Linear(block_dim, block_dim),
            )
        elif activation_function == "leaky_relu":
            self.net = nn.Sequential(
                nn.Linear(block_dim, block_dim),
                nn.LeakyReLU(negative_slope = slope, inplace=True),
                nn.Linear(block_dim, block_dim),
                )
        else:
            sys.exit("Please provide valid activation function.\
                      Choose among 'relu' and 'leaky_relu'. \
                     'leaky_relu' slope defaults to 0.1")
        self.inner_norm = nn.LayerNorm(block_dim)
        self.outer_norm = nn.LayerNorm(block_dim)
    
    def forward(self, x):
        return self.outer_norm(x + self.inner_norm(x + self.net(x)))

class Generator(nn.Module):
    
    def __init__(self, n_layers, block_dim, activation_function = 'relu', slope = 0.1, snm = False):
        super().__init__()
        self.net = nn.Sequential(
            *[Block(block_dim, activation_function, slope, snm) for _ in range(n_layers)]
        )
        #self.last_mlp = Block(block_dim, activation_function, slope)

    def init_weights(self, activation_function = 'relu'):
        if isinstance(self, nn.Linear):
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0.01)
        else:
            pass

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    
    def __init__(self, n_layers, block_dim, activation_function = 'relu', slope = 0.1, snm = True):
        super().__init__()
        self.net = nn.Sequential(
            *[Block(block_dim, activation_function, slope, snm) for _ in range(n_layers)]
        )

    def init_weights(self, activation_function = 'relu'):
        if isinstance(self, nn.Linear):
            torch.nn.init.kaiming_normal_(self.weight, nonlinearity = activation_function)
            if self.bias is not None:
                self.bias.data.fill_(0.01)
        else:
            pass
        
    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()  

    def forward(self, x):
        return self.net(x)