from ae import AutoEncoder
from loss_functions import encoder_loss, reconstruction_loss
import torch
import random
import os
import sys
import math
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
                                   
def load_ae(config):
    print("Loading pretrained ae...")
    model_3 = 'epoch_5_model.pth'
    base_path = '/content/gdrive/MyDrive/ATGWRL/'
    saved_models_dir = os.path.join(base_path, r'saved_aes')
    model_3_path = os.path.join(saved_models_dir, model_3)
    model = AutoEncoder(config)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_3_path):
            model.load_state_dict(torch.load(model_3_path), strict = False)
        else:
            sys.exit("AE model path does not exist")
    else:
        sys.exit("AE path does not exist")

    return model
    
def inverse_sigmoid_schedule(i, k):
    a = k/(k + math.exp(i/k))
    return a
    
def validation_set_acc(config, model, val_set):
    re_list = []
    for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, val_set)):
        original_lens_batch = real_lengths(batch)
        padded_batch = pad_batch(batch)
        targets = pad_batch_and_add_EOS(batch)
        weights = return_weights(original_lens_batch)

        weights = torch.FloatTensor(weights).to(model.device)
        padded_batch = torch.LongTensor(padded_batch).to(model.device)
        targets = torch.LongTensor(targets).to(model.device)
        
        with torch.no_grad():
            decoded_logits = model(padded_batch, original_lens_batch)

            reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
            
        re_list.append(reconstruction_error.item())
        
    val_error = sum(re_list) / len(re_list)
    print("val_error", val_error)
    
    return val_error

def train(config, 
          num_epochs = 7,
          logging_interval = 100, 
          saving_interval = 5000,
          plotting_interval = 2000,
          enc_loss_lambda = 0.2,
          validation_size = 10_000):
    
    data_path = "corpus_v20k_ids.txt"
    vocab_path = "vocab_20k.txt"
    print("loading data: {} and vocab: {}".format(data_path, vocab_path))
    
    reduced_data_size = 600_000
    print("reduced_data_size", reduced_data_size)
    
    data = load_data_from_file(data_path, reduced_data_size + validation_size)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))
    vocab, revvocab = load_vocab(vocab_path)
    config.vocab_size = len(revvocab)
    random.seed(10)
    kl_annealing_iters = 20_000
    kl_annealing_iters = kl_annealing_iters // (config.ae_batch_size / 100)

    model = AutoEncoder(config)
    model = model.apply(AutoEncoder.init_weights)
    model.to(model.device)
    #print("reloading model 5")
    #model = load_ae(config)
    #num_epochs = 2
    # print("changed epochs to ", num_epochs)
    sigmoid_rate = (data_len * num_epochs / config.ae_batch_size) / 23

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
    
    print("Starting AE training. Number of training epochs {}, batch_size {}".format(num_epochs, config.ae_batch_size))
    print("Logging interval: ", logging_interval)

    #if kl_annealing_iters is not None:
        #print("Peforming monotonic KL annealing for first {} batches".format(kl_annealing_iters))
        
        
    assert config.latent_dim == config.block_dim, "GAN block dimension and latent dimension must be equal"

    kl_annealing_counter = 0
    re_list = []
    encoder_loss_list = []
    kl_div_list = []
    kl_weight_list = []
    
    print("enc_loss_lambda {}".format(enc_loss_lambda))
    val_error_per_epoch = []
    
    print("encoder dim {}, decoder dim {}, latent_dim {}, input dropout {}, lr {}, embedding size {}, vocab size {}, batch size {}".format(config.encoder_dim, config.decoder_dim, config.latent_dim, config.dropout_prob, config.ae_learning_rate, config.word_embedding, config.vocab_size, config.ae_batch_size))

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, all_data)):
            kl_weight = 1 if kl_annealing_counter > kl_annealing_iters else kl_annealing_counter / kl_annealing_iters 
            kl_annealing_counter += 1
            
            #teacher_force_prob = inverse_sigmoid_schedule(kl_annealing_counter, sigmoid_rate)
            teacher_force_prob = 0

            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            targets = pad_batch_and_add_EOS(batch)
            weights = return_weights(original_lens_batch)

            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)
            targets = torch.LongTensor(targets).to(model.device)

            with torch.cuda.amp.autocast():

                decoded_logits, z, re_embedded = model(padded_batch, original_lens_batch, teacher_force_prob)

                reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                e_loss = encoder_loss(model, z, re_embedded, original_lens_batch)
                
                loss = (1 - enc_loss_lambda) * reconstruction_error + enc_loss_lambda * e_loss 
                
                re_list.append(reconstruction_error.item())
                encoder_loss_list.append(e_loss.item() * 5)
                #kl_div_list.append(e_loss.item() * 50)
                #kl_weight_list.append(0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(reconstruction_error.item())

            if batch_idx % logging_interval == 0:
                progress = ((batch_idx+1) * config.ae_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                progress = progress * 100
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | Encoder loss {:.10f}'.format(progress, epoch_idx, batch_idx, loss.item(), reconstruction_error.item(), e_loss.item()))

            if kl_annealing_counter % saving_interval == 0 and kl_annealing_counter > 0:
                print("Progress: {}, saving model...".format(progress))
                save_model(epoch_idx, model)
                
            if kl_annealing_counter > 0 and kl_annealing_counter % plotting_interval == 0:
                my_plot(len(re_list), re_list, encoder_loss_list)
                    
        #val_error = validation_set_acc(config, model, val)
        #val_error_per_epoch.append(val_error)
                    
    print("Training complete, saving model")
    #print("val error per epoch")
    #for i in range(len(val_error_per_epoch)):
        #print(val_error_per_epoch[i])
    #epoch_idx = num_epochs
    #epoch_idx = 5
    save_model(epoch_idx, model)
    my_plot(len(re_list), re_list, encoder_loss_list)
    return log_dict