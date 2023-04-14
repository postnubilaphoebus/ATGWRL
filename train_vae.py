from models import VariationalAutoEncoder
from cnn_vae import CNNVariationalAutoEncoder
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
                                   matrix_from_pretrained_embedding, \
                                   load_vocab

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_ae_accs_to_file(model_name, train_regime, epoch_num, train_error, val_error, val_bleu):
    with open("ae_results.txt", "a") as f:
        f.write("\n")
        f.write("##################################################################################################################################" + "\n")
        f.write("model: {} train_regime: {} \n".format(model_name, train_regime))
        f.write("epoch_num: {}, train error {}, val error {}, val_bleu {} \n".format(epoch_num, train_error, val_error, val_bleu))
        f.write("##################################################################################################################################" + "\n" + "\n")
    f.close()

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
          num_epochs = 20,
          data_path = "corpus_v40k_ids.txt",
          vocab_path = "vocab_40k.txt", 
          logging_interval = 5, 
          saving_interval = 10_000,
          plotting_interval = 10_000,
          validation_size = 10_000,
          random_seed = 42):
    
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    kl_div_factor = 10.0
    print("kl_div_factor", kl_div_factor)
    latent_mode = "sparse" # choose dropout otherwise
    print("latent_mode", latent_mode)
    sparsity_penalty = 0.1

    print("loading data: {} and vocab: {}".format(data_path, vocab_path)) 
    data = load_data_from_file(data_path, 1010_000)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))
    vocab, revvocab = load_vocab(vocab_path, 40_000)
    config.vocab_size = len(revvocab)

    if latent_mode == "sparse":
        config.use_dropout = False
    else:
        config.use_dropout = True

    config.pretrained_embedding = True
    config.word_embedding = 100
    config.encoder_dim = 600
    config.ae_batch_size = 128
    train_regimes = ["normal", "word deletion", "masking", "word-mixup"]

    if config.pretrained_embedding == True:
        assert config.word_embedding == 100, "glove embedding can only have dim 100, change config"
        glove = torchtext.vocab.GloVe(name='twitter.27B', dim=100) # 27B is uncased
        weights_matrix = matrix_from_pretrained_embedding(list(vocab.keys()), config.vocab_size, config.word_embedding, glove)
    else:
        weights_matrix = None

    model = VariationalAutoEncoder(config, weights_matrix)
    #model = CNNVariationalAutoEncoder(config, weights_matrix)
    model = model.apply(VariationalAutoEncoder.init_weights)
    model = model.to(model.device)

    # regularise latent dimension:
    # to_latent.weight

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

    print("######################################################")
    print("######################################################")

    for epoch_idx in range(num_epochs):
        epoch_wise_loss = []
        for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, all_data)):
            iter_counter += 1
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

                if latent_mode == "sparse":
                    l1_sum = model.to_latent.weight.abs().sum()
                    l1_regularization = (l1_sum * sparsity_penalty / config.latent_dim)

                loss = reconstruction_error+ kl_div_factor * kl_div + l1_regularization
                print("reconstruction_error {}, kl_div * kl_div_factor {}, lr reg{}".format(reconstruction_error.item(), kl_div.item() * kl_div_factor, l1_regularization.item()))
                
                re_list.append(reconstruction_error.item())
                epoch_wise_loss.append(reconstruction_error.item())
                kl_list.append(kl_div.item())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
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
                
        save_model(epoch_idx+1, model, train_regimes[0], latent_mode)
        if validation_size >= config.ae_batch_size:
            val_error, bleu_score = validation_set_acc(config, model, val, revvocab)
        write_ae_accs_to_file(model.name, train_regimes[0], epoch_idx+1, sum(epoch_wise_loss) / len(epoch_wise_loss), val_error, bleu_score)

    return log_dict