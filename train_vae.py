from models import VariationalAutoEncoder
from loss_functions import reconstruction_loss, validation_set_acc
import torch
import random
import numpy as np
import warnings
import torchtext
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   pad_batch_and_add_EOS, \
                                   average_over_nonpadded, \
                                   my_plot, \
                                   matrix_from_pretrained_embedding, \
                                   load_vocab, \
                                   autoencoder_info

def kl_loss(weights, z_mean_list, z_log_var_list):
    weights = torch.transpose(weights, 1, 0)
    kl_div = []
    for weight, z_mean, z_log_var in zip(weights, z_mean_list, z_log_var_list):
        kl_div_t = -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
        kl_div_t = torch.mean(kl_div_t, dim = 1)
        kl_div.append(weight * kl_div_t)
    kl_div = torch.stack((kl_div))
    kl_div = torch.sum(kl_div, dim = 0) # sum over seqlen
    kl_div = average_over_nonpadded(kl_div, weights, 0) # average over seqlen
    return torch.mean(kl_div)

def train(config, 
          num_epochs = 5,
          data_path = "corpus_v20k_ids.txt",
          vocab_path = "vocab_20k.txt", 
          logging_interval = 5, 
          saving_interval = 10_000,
          plotting_interval = 10_000,
          validation_size = 10_000,
          random_seed = 42):
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print("loading data: {} and vocab: {}".format(data_path, vocab_path)) 
    data = load_data_from_file(data_path, 10_000_000)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))
    vocab, revvocab = load_vocab(vocab_path)
    config.vocab_size = len(revvocab)

    model = VariationalAutoEncoder(config)
    model = model.apply(VariationalAutoEncoder.init_weights)
    model = model.to(model.device)

    optimizer = torch.optim.Adam(lr = config.ae_learning_rate, 
                                 params = model.parameters(),
                                 betas = (config.ae_betas[0], config.ae_betas[1]),
                                 eps=1e-08)
                                
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []} 
        
    model.train()

    print("######################################################")
    print("######################################################")
    print("Starting VAE training. Number of training epochs: {}".format(num_epochs))
    print("Logging interval:", logging_interval)
    assert config.latent_dim == config.block_dim, "GAN block dimension and latent dimension must be equal"
    iter_counter = 0
    re_list = []
    kl_list = []

    config.ae_batch_size = 50
    print("######################################################")
    print("######################################################")

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, all_data)):
            #kl_weight = 1 if iter_counter > kl_annealing_iters else iter_counter / kl_annealing_iters 
            iter_counter += 1
            
            #teacher_force_prob = inverse_sigmoid_schedule(iter_counter, sigmoid_rate)
            #teacher_force_prob = 0

            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            targets = pad_batch_and_add_EOS(batch)
            weights = return_weights(original_lens_batch)

            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)
            targets = torch.LongTensor(targets).to(model.device)

            with torch.cuda.amp.autocast():

                z_mean_list, z_log_var_list, decoded_logits = model(padded_batch, original_lens_batch)

                reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)

                kl_div = kl_loss(weights, z_mean_list, z_log_var_list)
                
                loss = reconstruction_error + kl_div
                
                re_list.append(reconstruction_error.item())
                kl_list.append(kl_div.item())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(reconstruction_error.item())

            if iter_counter > 0 and iter_counter % logging_interval == 0:
                progress = ((batch_idx+1) * config.ae_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                progress = progress * 100
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | current lr: {:.6f} | kl divergence {:.4f}'\
                    .format(progress,epoch_idx, batch_idx+1, loss.item(), reconstruction_error.item(), optimizer.param_groups[0]['lr'], kl_div.item()))
                    
    #print("Training complete, saving model")
    epoch_idx = num_epochs
    save_model(epoch_idx, model)
    #my_plot(len(re_list), re_list)
    if validation_size >= config.ae_batch_size:
        val_error, bleu_score = validation_set_acc(config, model, val, revvocab)
    #config_performance_cnn(config, label_smoothing, bleu_score, val_error, model_name)
    #config_performance(config, label_smoothing, bleu_score, val_error, model_name)

    return log_dict