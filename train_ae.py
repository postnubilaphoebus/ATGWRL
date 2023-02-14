from ae import AutoEncoder
from loss_functions import encoder_loss, reconstruction_loss
import torch
import random
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   pad_batch_and_add_EOS, \
                                   my_plot, \
                                   create_bpe_tokenizer, \
                                   tokenize_data, \
                                   load_vocab

def train(config, 
          num_epochs = 5,
          logging_interval = 100, 
          saving_interval = 5000,
          plotting_interval = 10_000,
          enc_loss_lambda = 0.2,
          validation_size = 10_000):
    
    data_path = "corpus_v20k_ids.txt"
    vocab_path = "vocab_20k.txt"
    print("loading data: {} and vocab: {}".format(data_path, vocab_path))
    
    data = load_data_from_file(data_path)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))
    vocab, revvocab = load_vocab(vocab_path)
    config.vocab_size = len(revvocab)
    random.seed(10)
    kl_annealing_iters = 20_000
    kl_annealing_iters = kl_annealing_iters // (config.batch_size / 100)

    model = AutoEncoder(config)
    model = model.apply(AutoEncoder.init_weights)
    model.to(model.device)

    optimizer = torch.optim.Adam(lr = config.ae_learning_rate, 
                                 params = model.parameters(),
                                 betas = (0.9, 0.999),
                                 eps=1e-08)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []} 
        
    model.train()
    
    print("Starting AE training. Number of training epochs {}, batch_size {}".format(num_epochs, config.batch_size))
    print("Logging interval: ", logging_interval)

    if kl_annealing_iters is not None:
        print("Peforming monotonic KL annealing for first {} batches".format(kl_annealing_iters))

    kl_annealing_counter = 0
    re_list = []
    kl_div_list = []
    kl_weight_list = []
    
    print("enc_loss_lambda {}".format(enc_loss_lambda))
    
    print("encoder dim {}, decoder dim {}, input dropout {}, lr {}, embedding size {}, vocab size {}, batch size {}".format(config.encoder_dim, config.decoder_dim, config.dropout_prob, config.ae_learning_rate, config.word_embedding, config.vocab_size, config.batch_size))

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.batch_size, all_data)):
            kl_weight = 1 if kl_annealing_counter > kl_annealing_iters else kl_annealing_counter / kl_annealing_iters 
            kl_annealing_counter += 1

            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            targets = pad_batch_and_add_EOS(batch)
            weights = return_weights(original_lens_batch)

            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)
            targets = torch.LongTensor(targets).to(model.device)

            with torch.cuda.amp.autocast():

                decoded_logits, encoded, re_embedded = model(padded_batch, original_lens_batch)

                reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                e_loss = encoder_loss(model, encoded, re_embedded, original_lens_batch)
                
                loss = (1 - enc_loss_lambda) * reconstruction_error + enc_loss_lambda * e_loss
                
                re_list.append(reconstruction_error.item())
                kl_div_list.append(e_loss.item() * 50)
                kl_weight_list.append(0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(reconstruction_error.item())

            if batch_idx % logging_interval == 0:
                progress = ((batch_idx+1) * config.batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | Encoder Loss {:.4f}'.format(progress, epoch_idx, batch_idx, loss.item(), reconstruction_error.item(), e_loss.item()))

            if kl_annealing_counter % saving_interval == 0 and kl_annealing_counter > 0:
                print("Progress: {}, saving model...".format(progress))
                save_model(epoch_idx, model)
                if kl_annealing_counter % plotting_interval == 0:
                    my_plot(len(re_list), re_list, kl_div_list, kl_weight_list)

    print("Training complete, saving model")
    epoch_idx = num_epochs
    save_model(epoch_idx, model)
    my_plot(len(re_list), re_list, kl_div_list, kl_weight_list)
    return log_dict