from vae import VariationalAutoEncoder
import torch
import random
import torch.nn.functional as F
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from utils.helper_functions import load_data_and_create_vocab, \
                                   prepare_data, \
                                   yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   average_over_nonpadded, \
                                   reformat_decoded_batch

def load_first_two_lines(data_path):
    print("loading one line for overfitting")
    data = []
    data_file = open(data_path, 'r')
    idx = 0

    while True:
        line = data_file.readline()
        line = line[1:-2]
        line = line.replace(" ", "")
        line = line.split(",")
        line = [int(x) for x in line]
        data.append(line)
        idx += 1
        if idx > 1:
            break
    data_file.close()
    return data
    
def my_plot(epochs, re_list, kl_div_list, kl_weight_list):
    re_list = np.array(re_list)
    kl_div_list = np.array(kl_div_list)
    kl_weight_list = np.array(kl_weight_list)

    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = 'Plotted loss after ' + str(epochs) + 'batches.png'
    final_directory = os.path.join(directory, filename)

    temp = epochs + 1
    epochs = []
    for i in range(temp):
        epochs.append(i)

    epochs = np.array(epochs)
    
    plt.plot(epochs, re_list, label = 'reconstruction error')
    plt.plot(epochs, kl_div_list, label = 'KL divergence')
    plt.plot(epochs, kl_weight_list, label = 'kl weight')
    
    plt.yscale('log')
    plt.xlabel('Batchs')
    plt.ylabel('loss values (log)')
    plt.title('Loss plotted over ' + str(temp) + ' batches')
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

def overfit(config, 
          loss_fn = None, 
          num_epochs = 50,
          logging_interval = 100, 
          saving_interval = 5000,
          kl_annealing_iters = 25_000,
          kl_scaling = 0.1,
          enc_loss_lambda = 0.01):
              
    use_amp = True

    vocab, revvocab, dset = load_data_and_create_vocab()
    config.vocab_size = len(revvocab)
    _ = prepare_data(dset, vocab)
    random.seed(10)
    
    two_lines = load_first_two_lines("bookcorpus_ids.txt")

    model = VariationalAutoEncoder(config)
    model.to(model.device)

    optimizer = torch.optim.Adam(lr = config.vae_learning_rate, 
                                 params = model.parameters(),
                                 betas = (0.9, 0.999),
                                 eps=1e-08)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss(reduction='none')
        cross_ent = torch.nn.CrossEntropyLoss(reduction='none')
        
    model.train()
    
    print("start overfitting...")

    kl_annealing_counter = 0
    kl_annealing_iters = 20_000

    train_iters = 25_000
    config.batch_size = 2
    
    re_list = []
    kl_div_list = []
    #encoder_loss_list = []
    kl_weight_list = []

    for train_idx in range(train_iters):
        kl_weight = 1 if kl_annealing_counter > kl_annealing_iters else kl_annealing_counter / kl_annealing_iters
        kl_annealing_counter += 1
        original_lens_batch = real_lengths(two_lines)
        padded_batch = pad_batch(two_lines)
        weights = return_weights(original_lens_batch)
        weights = torch.FloatTensor(weights).to(model.device)
        padded_batch = torch.LongTensor(padded_batch).to(model.device)

        with torch.cuda.amp.autocast():
            encoded, z_mean, z_log_var, decoded_tokens, decoded_logits, time_list = model(padded_batch, original_lens_batch, True)
                
            # as given by Oshri and Khandwala in: There and Back Again: Autoencoders for Textual Reconstruction:
            #_, _, decoded_enc_hidden = model.autoregressive_decoding(encoded, True) # D(E(x))
            #encoded_dec_hidden, _ = model.encoder(decoded_enc_hidden) # E(D(E(x)))
            #encoded_dec_hidden = encoded_dec_hidden[-1, :, :] #take last hidden state
                
            #encoded_dec_hidden = torch.transpose(encoded_dec_hidden, 1, 0) # [batch, encoder_dim] -> [encoder_dim, batch]
            #encoded = torch.transpose(encoded, 1, 0) # [batch, encoder_dim] -> [encoder_dim, batch]
            #encoder_loss = []
                
            #for dim_enc, dim_enc_dec in zip(encoded, encoded_dec_hidden):
                #encoder_loss.append(loss_fn(dim_enc_dec, dim_enc))
                    
            #encoder_loss = torch.stack((encoder_loss))
            #encoder_loss = torch.mean(encoder_loss)
                
            kl_div = - 0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
            kl_div = torch.mean(kl_div)
            kl_div = torch.clip(kl_div, min = 0)

            reconstruction_error = []

            weights = torch.transpose(weights, 1, 0)
            padded_batch = torch.transpose(padded_batch, 1, 0)
            padded_batch = torch.nn.functional.one_hot(padded_batch, num_classes = config.vocab_size).to(torch.float32)

            for weight, target, logit in zip(weights, padded_batch, decoded_logits):
                ce_loss = cross_ent(logit, target)
                ce_loss = torch.mean(ce_loss, dim = -1)
                reconstruction_error.append(ce_loss * weight)

            reconstruction_error = torch.stack((reconstruction_error))
            reconstruction_error = torch.sum(reconstruction_error, dim = 0) # sum over seqlen
            reconstruction_error = average_over_nonpadded(reconstruction_error, weights, 0) # av over seqlen
            reconstruction_error = torch.mean(reconstruction_error) # mean over batch
                
            loss = reconstruction_error + kl_weight * kl_div * kl_scaling #+ encoder_loss * enc_loss_lambda
            re_list.append(reconstruction_error.item())
            kl_div_list.append(kl_div.item())
            #encoder_loss_list.append(encoder_loss.item() * enc_loss_lambda)
            kl_weight_list.append(kl_weight)
            
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if train_idx % logging_interval == 0:
            progress = train_idx / train_iters
            print('Progress {:.4f}% | Loss {:.10f} | Reconstruction Error {:.10f} | KL_div {:.10f} | KL_weight {:.5f} | KL_scaling {}'.format(progress, loss.item(), reconstruction_error.item(), kl_div.item(), kl_weight, kl_scaling))

    print("Training complete")
    my_plot(train_idx, re_list, kl_div_list, kl_weight_list)
    



