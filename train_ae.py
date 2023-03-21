from models import AutoEncoder, CNNAutoEncoder, ExperimentalAutoencoder
from loss_functions import reconstruction_loss, validation_set_acc
import torch
import random
import numpy as np
import sys
import scipy
import warnings
import torchtext
from sklearn.metrics.pairwise import cosine_similarity
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   pad_batch_and_add_EOS, \
                                   my_plot, \
                                   matrix_from_pretrained_embedding, \
                                   load_vocab, \
                                   autoencoder_info

def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1
    return lo

def sample_word(cumsummed, normalised, original_word):
    max_for_word = cumsummed[original_word][-1]
    sampled_val = random.uniform(0, max_for_word)
    sampled_word = bisect_right(cumsummed[original_word], sampled_val)
    return sampled_word

def proportional_prob(config, weights_matrix, percentile_cutoff = 10.0):
    similarities = cosine_similarity(weights_matrix)
    normalised = np.array([(x+1) / 2 for x in similarities])
    np.fill_diagonal(normalised, 0)
    perc = np.percentile(normalised, percentile_cutoff, axis = 0)
    for row in range(config.vocab_size):
        for col in range(config.vocab_size):
            normalised[row, col] = 0 if normalised[row, col] < perc[row] else normalised[row, col]
    cumsummed = np.cumsum(normalised, axis = 1)
    return cumsummed, normalised

def config_performance(config, label_smoothing, bleu4, val_loss, model_name):
    with open("ae_results.txt", "a") as f:
        f.write("##################################################################################################################################" + "\n")
        f.write("model name: {}, lr: {}, attn: {}, drop: {}, layer_norm: {}, lbl_smooth: {}, enc_dim: {}"
                .format(model_name,
                        str(config.ae_learning_rate), 
                        str(config.attn_bool), 
                        str(config.dropout_prob), 
                        str(config.layer_norm),
                        str(label_smoothing), 
                        str(config.encoder_dim)))
        f.write("\n")
        f.write("Bleu4: {}, Validation loss: {}".format(bleu4, val_loss))
        f.write("\n")
        f.write("##################################################################################################################################" + "\n" + "\n")
        f.close()

def config_performance_cnn(config, label_smoothing, bleu4, val_loss, model_name):
    with open("ae_cnn_results.txt", "a") as f:
        f.write("##################################################################################################################################" + "\n")
        f.write("model name {}, lr {}, drop {}, kernel1 {}, kernel2 {}, out channels {}, label smoothing {}"
                .format(model_name,
                        str(config.ae_learning_rate),
                        str(config.dropout_prob),
                        str(config.kernel_sizes[0]),
                        str(config.kernel_sizes[1]),
                        str(config.out_channels),
                        str(label_smoothing))) 
        f.write("\n")
        f.write("Bleu4: {}, Validation loss: {}".format(bleu4, val_loss))
        f.write("\n")
        f.write("##################################################################################################################################" + "\n" + "\n")
        f.close()

def train(config, 
          num_epochs = 7,
          model_name = "cnn_autoencoder",
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
    data = load_data_from_file(data_path, 110_000)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))
    vocab, revvocab = load_vocab(vocab_path)
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
            #kl_weight = 1 if iter_counter > kl_annealing_iters else iter_counter / kl_annealing_iters 
            iter_counter += 1
            
            #teacher_force_prob = inverse_sigmoid_schedule(iter_counter, sigmoid_rate)
            teacher_force_prob = 1.0

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
                    
    #print("Training complete, saving model")
    #save_model(epoch_idx+1, model)
    #my_plot(len(re_list), re_list)
    #if validation_size >= config.ae_batch_size:
        #val_error, bleu_score = validation_set_acc(config, model, val, revvocab)
    #config_performance_cnn(config, label_smoothing, bleu_score, val_error, model_name)
    #config_performance(config, label_smoothing, bleu_score, val_error, model_name)

    return log_dict