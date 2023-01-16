from vae import VariationalAutoEncoder
import torch
from torch import nn
import utils.config as config
from utils.helper_functions import load_data_and_create_vocab, \
                                   prepare_data, \
                                   yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model

import random
import torch.nn.functional as F
import numpy as np

# monotonic annealing schedule for KL term

def train(config, 
          loss_fn = None, 
          num_epochs = 5,
          logging_interval = 100, 
          saving_interval = 500_000,
          kl_annealing_iters = 20_000):

    vocab, revvocab, dset = load_data_and_create_vocab()
    config.vocab_size = len(revvocab)
    _ = prepare_data(dset, vocab)
    random.seed(10)

    all_data = load_data_from_file("bookcorpus_ids.txt")

    model = VariationalAutoEncoder(config)
    model.to(model.device)

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
    
    print("Starting VAE training. Number of training epochs {}, batch_size {}".format(num_epochs, config.batch_size))
    print("Logging interval: ", logging_interval)
    if kl_annealing_iters is not None:
        print("Peforming monotonic KL annealing for first {} batches".format(kl_annealing_iters))

    data_len = len(all_data)

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.batch_size, all_data)):
            kl_weight = 1 if (batch_idx > kl_annealing_iters or epoch_idx > 0) else (batch_idx / kl_annealing_iters)

            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            max_sentlen = max(original_lens_batch)
            weights = return_weights(padded_batch, max_sentlen)

            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)

            encoded, z_mean, z_log_var, decoded_tokens, decoded_logits = model(padded_batch, original_lens_batch)

            kl_div = - 0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var))

            reconstruction_error = []

            weights = torch.transpose(weights, 1, 0)
            padded_batch = torch.transpose(padded_batch, 1, 0)
            padded_batch = torch.nn.functional.one_hot(padded_batch, num_classes = config.vocab_size).to(torch.float32)

            for weight, target, logit in zip(weights, padded_batch, decoded_logits):
                mse_loss = loss_fn(logit, target)
                reconstruction_error.append(mse_loss * 1)

            kl_div = torch.mean(kl_div)

            reconstruction_error = torch.stack((reconstruction_error))
            reconstruction_error = torch.mean(reconstruction_error)

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
                progress = epoch_idx / num_epochs + batch_idx / data_len
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | KL_div {:.10f} | KL_weight {:.5f}'.format(progress, epoch_idx, batch_idx, loss.item(), reconstruction_error.item(), kl_div.item(), kl_weight))

            if batch_idx % saving_interval == 0:
                print("Progress: {}, saving model...".format(progress))
                save_model(epoch_idx, model)

    print("Training complete, saving model")
    save_model(epoch_idx, model)
    return log_dict


