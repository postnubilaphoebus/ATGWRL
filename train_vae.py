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
                                   average_over_nonpadded, \
                                   reformat_decoded_batch

def model_usage_by_layer(time_list):
    total_time = sum(time_list)

    decoding_time = time_list[0] / total_time
    reparam_time = time_list[1] / total_time
    code_layer_time = time_list[2] / total_time
    encoder_plus_dropout_time = time_list[3] / total_time
    embedding_time = time_list[4] / total_time

    print("Model times by percentage: Embedding {} | Encoder {} | Code Layer {} | Reparam {} | Decoding {}".format(embedding_time, encoder_plus_dropout_time, code_layer_time, reparam_time, decoding_time))
    
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


def train(config, 
          loss_fn = None, 
          num_epochs = 50,
          logging_interval = 100, 
          saving_interval = 5000,
          kl_annealing_iters = 10_000,
          kl_scaling = 0.0,
          enc_loss_lambda = 0.001):
              
    use_amp = True

    vocab, revvocab, dset = load_data_and_create_vocab()
    config.vocab_size = len(revvocab)
    _ = prepare_data(dset, vocab)
    random.seed(10)
    
    kl_annealing_iters = kl_annealing_iters // (config.batch_size / 100)

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
                'train_kl_loss_per_batch': []} #'train_encoder_loss_per_batch':[],

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss(reduction='none')
        cross_ent = torch.nn.CrossEntropyLoss(reduction='none')
        
    model.train()
    
    print("Starting VAE training. Number of training epochs {}, batch_size {}".format(num_epochs, config.batch_size))
    print("Logging interval: ", logging_interval)
    if kl_annealing_iters is not None:
        print("Peforming monotonic KL annealing for first {} batches".format(kl_annealing_iters))

    data_len = len(all_data)
    loss_calc_sum = 0
    model_time_sum = 0

    model_times_breakdown = [0, 0, 0, 0, 0]
    
    kl_annealing_counter = 0
    
    re_list = []
    kl_div_list = []
    #encoder_loss_list = []
    kl_weight_list = []
    
    print("encoder dim {}, decoder dim {}, input dropout {}, lr {}, embedding size {}, vocab size {}, batch size {}".format(config.encoder_dim, config.decoder_dim, config.dropout_prob, config.vae_learning_rate, config.word_embedding, config.vocab_size, config.batch_size))

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.batch_size, all_data)):
            kl_weight = 1 if kl_annealing_counter > kl_annealing_iters else kl_annealing_counter / kl_annealing_iters 
            kl_annealing_counter += 1

            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            weights = return_weights(original_lens_batch)

            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)

            with torch.cuda.amp.autocast():
                
                model_time_0 = time.time()

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

                model_time_1  = time.time()
                model_time = model_time_1 - model_time_0
                loss_calc_0 = time.time()
                
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
                loss_calc_1 = time.time()
                
            loss_calc_time = loss_calc_1 - loss_calc_0

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            loss_calc_sum += loss_calc_time
            model_time_sum += model_time

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(reconstruction_error.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            #log_dict['train_encoder_loss_per_batch'].append(encoder_loss.item())
            
            total_time = loss_calc_sum + model_time_sum

            if batch_idx % logging_interval == 0:
                progress = ((batch_idx+1) * config.batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | KL_div {:.10f} | KL_weight {:.5f} | KL_scaling {}'.format(progress, epoch_idx, batch_idx, loss.item(), reconstruction_error.item(), kl_div.item(), kl_weight, kl_scaling))

            if kl_annealing_counter % saving_interval == 0 and kl_annealing_counter > 0:
                print("Progress: {}, saving model...".format(progress))
                print("Model time {}%, loss_calc {}%".format(model_time_sum/total_time, loss_calc_sum/total_time))
                model_usage_by_layer(time_list)
                save_model(epoch_idx, model)

    print("Training complete, saving model")
    epoch_idx = num_epochs
    save_model(epoch_idx, model)
    my_plot(len(re_list), re_list, kl_div_list, kl_weight_list)
    return log_dict


