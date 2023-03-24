from models import AutoEncoder, CNNAutoEncoder, ExperimentalAutoencoder
from loss_functions import reconstruction_loss, validation_set_acc
import torch
import random
import numpy as np
import sys
import scipy
import warnings
import torchtext
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   pad_batch_and_add_EOS, \
                                   matrix_from_pretrained_embedding, \
                                   load_vocab, \
                                   autoencoder_info

def train(config, 
          num_epochs = 5,
          model_name = "cnn_autoencoder",
          data_path = "corpus_v40k_ids.txt",
          vocab_path = "vocab_40k.txt", 
          logging_interval = 100, 
          saving_interval = 10_000,
          plotting_interval = 10_000,
          validation_size = 10_000,
          random_seed = 42):
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print("loading data: {} and vocab: {}".format(data_path, vocab_path)) 
    data = load_data_from_file(data_path)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))
    vocab, revvocab = load_vocab(vocab_path, 40_000)
    config.vocab_size = len(revvocab)

    config.pretrained_embedding = True
    config.word_embedding = 100
    config.encoder_dim = 100
    config.ae_batch_size = 128
    if config.pretrained_embedding == True:
        assert config.word_embedding == 100, "glove embedding can only have dim 100, change config"
        glove = torchtext.vocab.GloVe(name='twitter.27B', dim=100) # 27B is uncased
        weights_matrix = matrix_from_pretrained_embedding(list(vocab.keys()), config.vocab_size, config.word_embedding, glove)
    else:
        weights_matrix = None

    if model_name == "default_autoencoder":
        model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)
    elif model_name == "cnn_autoencoder":
        model = CNNAutoEncoder(config, weights_matrix)
        model = model.apply(CNNAutoEncoder.init_weights)
        model.to(model.device)
    elif model_name == "experimental_autoencoder":
        model = ExperimentalAutoencoder(config, weights_matrix)
        model = model.apply(ExperimentalAutoencoder.init_weights)
        model.to(model.device)
    else:
        warnings.warn("Provided invalid model name. Loading default autoencoder...")
        model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)

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
    print("Starting AE training. Number of training epochs: {}".format(num_epochs))
    print("Logging interval:", logging_interval)
    assert config.latent_dim == config.block_dim, "GAN block dimension and latent dimension must be equal"
    iter_counter = 0
    re_list = []

    autoencoder_info(model, config)
    print("######################################################")
    print("######################################################")

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, all_data)):
            
            iter_counter += 1
            teacher_force_prob = 0
            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            targets = pad_batch_and_add_EOS(batch)
            weights = return_weights(original_lens_batch)
            weights = torch.FloatTensor(weights).to(model.device)
            padded_batch = torch.LongTensor(padded_batch).to(model.device)
            targets = torch.LongTensor(targets).to(model.device)

            with torch.cuda.amp.autocast():

                decoded_logits = model(padded_batch, original_lens_batch, teacher_force_prob)
                reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
                loss = reconstruction_error 
                re_list.append(reconstruction_error.item())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(reconstruction_error.item())

            if iter_counter > 0 and iter_counter % logging_interval == 0:
                progress = ((batch_idx+1) * config.ae_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                progress = progress * 100
                print('Progress {:.4f}% | Epoch {} | Batch {} | Loss {:.10f} | Reconstruction Error {:.10f} | current lr: {:.6f}'\
                    .format(progress,epoch_idx, batch_idx+1, loss.item(), reconstruction_error.item(), optimizer.param_groups[0]['lr']))
                
        save_model(epoch_idx+1, model)
        if validation_size >= config.ae_batch_size:
            _, _ = validation_set_acc(config, model, val, revvocab)
                    
    return log_dict