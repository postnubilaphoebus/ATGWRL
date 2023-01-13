from vae import VariationalAutoEncoder
import torch
from torch import nn
import utils.config as config
from utils.helper_functions import load_data_and_create_vocab, prepare_data, yieldBatch, load_data_from_file
import random
import torch.nn.functional as F
import numpy as np

# monotonic annealing schedule for KL term

def real_lengths(unpadded_list):
    sent_lens = [len(i) for i in unpadded_list]
    return sent_lens

def pad_batch(batch):
    list_len = [len(i) for i in batch]
    max_len = max(list_len)
    padded_batch = []
    for element in batch:
        element.extend([0] * (max_len - len(element)))
        padded_batch.append(element)

    return padded_batch

def return_weights(padded_batch, max_len):
    batch_weights = []
    for element in padded_batch:
        try:
            first_zero = padded_batch.index(0)
            weights = [1] * first_zero + [0] * (max_len - len(element))
        except:
            weights = [1] * len(element)
        batch_weights.append(weights)
    return batch_weights

def train(config, 
          loss_fn = None, 
          logging_interval=100, 
          kl_annealing_iters = 100_000):

    vocab, revvocab, dset = load_data_and_create_vocab()
    config.vocab_size = len(revvocab)
    data_size = prepare_data(dset, vocab)
    random.seed(10)

    all_data = load_data_from_file("bookcorpus_ids.txt")

    model = VariationalAutoEncoder(config)

    optimizer = torch.optim.Adam(lr = config.vae_learning_rate, 
                                 params = model.parameters(),
                                 betas = (0.9, 0.999),
                                 eps=1e-08)

    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn is None:
        loss_fn = F.mse_loss
    model.train()

    # Donahue trains for 5 epochs
    num_epochs = 5
    print("Starting VAE training. Number of training epochs {}, batch_size {}".format(num_epochs, config.batch_size))

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.batch_size, all_data)):
            kl_weight = 1 if batch_idx > kl_annealing_iters else (batch_idx / kl_annealing_iters)

            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            max_sentlen = max(original_lens_batch)
            weights = return_weights(padded_batch, max_sentlen)

            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)

            encoded, z_mean, z_log_var, decoded_tokens, decoded_logits = model(padded_batch, original_lens_batch)

            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                        - z_mean**2 
                                        - torch.exp(z_log_var), 
                                        axis=1)

            kl_div = torch.mean(kl_div, dim = 0)# average over batch dimension  

            reconstruction_error = []
            decoded_logits = torch.transpose(decoded_logits, 1, 0)
            print("logits size {}, weights size {}, padded_batch size {}".format(decoded_logits.size(0), weights.size(0), padded_batch.size(0)))

            for weight, target, logit in zip(weights, padded_batch, decoded_logits):
                import pdb; pdb.set_trace()
                mse_loss = loss_fn(logit, target)
                reconstruction_error.append(mse_loss * weight)


            import pdb; pdb.set_trace()


            loss = reconstruction_error + kl_weight * kl_div
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(reconstruction_error.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())

            if batch_idx % logging_interval == 0:
                print('Epoch {} | Batch {} | Loss: {:.4f}'.format(epoch_idx, batch_idx, loss))

    return log_dict


