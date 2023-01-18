from vae import VariationalAutoEncoder
import torch
import random
import torch.nn.functional as F
import time
from utils.helper_functions import load_data_and_create_vocab, \
                                   prepare_data, \
                                   yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   average_over_nonpadded

def model_usage_by_layer(time_list):
    total_time = sum(time_list)

    # [decoding_time, reparam_time, code_layer_time, encoder_plus_dropout_time, embedding_time]
    decoding_time = time_list[0] / total_time
    reparam_time = time_list[1] / total_time
    code_layer_time = time_list[2] / total_time
    encoder_plus_dropout_time = time_list[3] / total_time
    embedding_time = time_list[4] / total_time

    print("Model times by percentage: Embedding {} | Encoder {} | Code Layer {} | Reparam {} | Decoding {}".format(embedding_time, encoder_plus_dropout_time, code_layer_time, reparam_time, decoding_time))


def train(config, 
          loss_fn = None, 
          num_epochs = 5,
          logging_interval = 100, 
          saving_interval = 5000,
          kl_annealing_iters = 20_000):

    use_amp = True

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

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss(reduction='none')

    model.train()
    
    print("Starting VAE training. Number of training epochs {}, batch_size {}".format(num_epochs, config.batch_size))
    print("Logging interval: ", logging_interval)
    if kl_annealing_iters is not None:
        print("Peforming monotonic KL annealing for first {} batches".format(kl_annealing_iters))

    data_len = len(all_data)
    loss_calc_sum = 0
    model_time_sum = 0

    model_times_breakdown = [0, 0, 0, 0, 0]

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.batch_size, all_data)):
            kl_weight = 1 if (batch_idx > kl_annealing_iters or epoch_idx > 0) else (batch_idx / kl_annealing_iters)

            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            weights = return_weights(original_lens_batch)

            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)

            with torch.cuda.amp.autocast():
                
                model_time_0 = time.time()

                encoded, z_mean, z_log_var, decoded_tokens, decoded_logits, time_list = model(padded_batch, original_lens_batch)
                
                model_time_1  = time.time()
                
                model_time = model_time_1 - model_time_0
                
                loss_calc_0 = time.time()
                

                kl_div = - 0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
                kl_div = torch.mean(kl_div)

                reconstruction_error = []

                weights = torch.transpose(weights, 1, 0)
                padded_batch = torch.transpose(padded_batch, 1, 0)
                padded_batch = torch.nn.functional.one_hot(padded_batch, num_classes = config.vocab_size).to(torch.float32)

                for weight, target, logit in zip(weights, padded_batch, decoded_logits):
                    mse_loss = loss_fn(logit, target)
                    mse_loss = torch.mean(mse_loss, dim = -1)
                    reconstruction_error.append(mse_loss * weight)

                reconstruction_error = torch.stack((reconstruction_error))
                reconstruction_error = torch.sum(reconstruction_error, dim = 0) # sum over seqlen
                reconstruction_error = average_over_nonpadded(reconstruction_error, weights, 0) # av over seqlen
                reconstruction_error = torch.mean(reconstruction_error) # mean over batch

                loss = reconstruction_error + kl_weight * kl_div
                
                loss_calc_1 = time.time()
                
            loss_calc_time = loss_calc_1 - loss_calc_0

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            loss_calc_sum += loss_calc_time
            model_time_sum += model_time
            model_times_breakdown = [sum(x) for x in zip(model_times_breakdown, time_list)]

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(reconstruction_error.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())

            if batch_idx % logging_interval == 0:
                progress = epoch_idx / num_epochs + batch_idx / data_len
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | KL_div {:.10f} | KL_weight {:.5f}'.format(progress, epoch_idx, batch_idx, loss.item(), reconstruction_error.item(), kl_div.item(), kl_weight))
                total_time = loss_calc_sum + model_time_sum
                print("Model time {}%, loss_calc {}%".format(model_time_sum/total_time, loss_calc_sum/total_time))
                model_usage_by_layer(time_list)

            if batch_idx % saving_interval == 0 and batch_idx > 0:
                print("Progress: {}, saving model...".format(progress))
                save_model(epoch_idx, model)

    print("Training complete, saving model")
    save_model(epoch_idx, model)
    return log_dict


