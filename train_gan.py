import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import os
import copy
import sys
from models import AutoEncoder, CNNAutoEncoder, Generator, Critic, VariationalAutoEncoder
from distribution_fitting import distribution_fitting, distribution_constraint
import random
import matplotlib.pyplot as plt
import time
import warnings
from utils.helper_functions import yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   save_gan, \
                                   cutoff_scores, \
                                   find_min_and_max, \
                                   plot_singular_values, \
                                   normalise, \
                                   re_scale, \
                                   sample_multivariate_gaussian, \
                                   sample_bernoulli, \
                                   plot_gan_acc, \
                                   plot_gan_loss, \
                                   sample_batch, \
                                   singular_values, \
                                   write_accs_to_file

def vae_encoding(vae, padded_batch, original_lens_batch):
    output = vae.encoder(padded_batch, original_lens_batch) #consumes 97% of function time
    output = torch.transpose(output, 1, 0)

    z_mean_list, z_log_var_list = vae.holistic_regularisation(output)

    # extract h_t-1
    z_mean_context = []
    z_log_var_context = []
    for z_mean_seq, z_log_var_seq, unpadded_len in zip(z_mean_list, z_log_var_list, original_lens_batch):
        z_mean_context.append(z_mean_seq[unpadded_len-1, :])
        z_log_var_context.append(z_log_var_seq[unpadded_len-1, :])
    
    z_mean_context = torch.stack((z_mean_context))
    z_log_var_context = torch.stack((z_log_var_context))
    z = vae.reparameterize(z_mean_context, z_log_var_context)

    return z

def load_ae(model_name, config):
    weights_matrix = None

    if model_name == "default_autoencoder":
        model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)
        model_5 = "epoch_10_model_default_autoencoder_regime_normal_latent_mode_dropout.pth"
        print("loading", model_5)
    elif model_name == "cnn_autoencoder":
        model = CNNAutoEncoder(config, weights_matrix)
        model = model.apply(CNNAutoEncoder.init_weights)
        model.to(model.device)
        model_5 = 'epoch_7_model_cnn_autoencoder_regime_normal_latent_mode_dropout.pth'
        print("loading", model_5)
    elif model_name == "variational_autoencoder":
        model = VariationalAutoEncoder(config, weights_matrix)
        model = model.apply(VariationalAutoEncoder.init_weights)
        model.to(model.device)
        model_5 = "epoch_20_model_variational_autoencoder_regime_normal_latent_mode_dropout.pth"
        print("loading", model_5)
    else:
        warnings.warn("Provided invalid model name. Loading default autoencoder...")
        model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)
        model_5 = "epoch_10_model_default_autoencoder_regime_normal_latent_mode_dropout.pth"
        print("loading", model_5)

    print("Loading pretrained ae of type {}, epoch 5...".format(model_name))
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_aes')
    model_5_path = os.path.join(saved_models_dir, model_5)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_5_path):
            model.load_state_dict(torch.load(model_5_path, map_location=torch.device(config.device)), strict = False)
        else:
            sys.exit("AE model path does not exist")
    else:
        sys.exit("AE path does not exist")

    return model
    
def load_gan(config):
    print("Loading pretrained generator...")
    print("loading epoch 10")
    model_15 = 'generator_epoch_10_model.pth'
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_gan')
    model_15_path = os.path.join(saved_models_dir, model_15)
    model = Generator(config.n_layers, config.block_dim)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_15_path):
            model.load_state_dict(torch.load(model_15_path, map_location=torch.device(config.device)), strict = False)
        else:
            sys.exit("GAN model path does not exist")
    else:
        sys.exit("GAN path does not exist")

    return model

def load_crit(config):
    print("Loading pretrained disc...")
    print("loading epoch 10")
    model_15 = 'critic_epoch_10_model.pth'
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_gan')
    model_15_path = os.path.join(saved_models_dir, model_15)
    model = Critic(config.n_layers, config.block_dim)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_15_path):
            model.load_state_dict(torch.load(model_15_path, map_location=torch.device(config.device)), strict = False)
        else:
            sys.exit("GAN model path does not exist")
    else:
        sys.exit("GAN path does not exist")

    return model

def compute_grad_penalty(config, critic, real_data, fake_data):
    B = real_data.size(0)
    alpha = torch.FloatTensor(np.random.random((B, 1))).to(config.device)
    sample = alpha * real_data + (1-alpha) * fake_data
    sample.requires_grad_(True)
    score = critic(sample)
    outputs = torch.FloatTensor(B, config.latent_dim).fill_(1.0)
    outputs.requires_grad_(False)
    outputs = outputs.to(config.device)
    grads = autograd.grad(
        outputs=score,
        inputs=sample,
        grad_outputs=outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_penalty = ((grads.norm(2, dim=1) - 1.) ** 2).mean()
    return grad_penalty

def train_gan(config,
              model_name =  "default_autoencoder",
              num_sents = 30_000,
              validation_size = 10_000,
              unroll_steps = 0,
              norm_data = False,
              gdf = False,
              gdf_scaling_factor = 1.0,
              num_epochs = 10,
              gp_lambda = 10,
              print_interval = 100,
              plotting_interval = 50_000, 
              n_times_critic = 5,
              data_path = "corpus_v40k_ids.txt", 
              vocab_path = "vocab_40k.txt"):
    
    config.vocab_size = 40_001

    if model_name == "variational_autoencoder":
        config.encoder_dim = 600
        config.word_embedding = 100
    else:
        config.encoder_dim = 100
        config.word_embedding = 100

    print("num_epochs", num_epochs)
    autoencoder = load_ae(model_name, config)
    autoencoder.eval()

    data = load_data_from_file(data_path, num_sents)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))

    if gdf == True:
        fitted_distribution = distribution_fitting(config, autoencoder, all_data)
        fitted_distribution = fitted_distribution.to(config.device)

    config.gan_batch_size = 512
    config.n_layers = 20
    crit_activation_function = "relu"
    gen_activation_function = "relu"
    config.c_learning_rate = 1e-4
    config.g_learning_rate = 1e-4

    print("batch size {}, block_dim {}".format(config.gan_batch_size, config.block_dim))
    print("nlayers critic {}, nlayers generator {}".format(config.n_layers, config.n_layers))
    print("n_times_critic", n_times_critic)
    print("activation G {}, activation C {}".format(gen_activation_function, crit_activation_function))
    print("unroll steps", unroll_steps)
    print("G lr", config.g_learning_rate)
    print("D lr", config.c_learning_rate)
    print("Adam betas {}, {}".format(config.gan_betas[0], config.gan_betas[1]))
    print("Using WGAN with spectral norm (WGAN-SN)")

    gen = Generator(config.n_layers, config.block_dim, gen_activation_function, snm = False).to(config.device)
    crit = Critic(config.n_layers, config.block_dim, crit_activation_function, snm = True).to(config.device)
    gen = gen.apply(Generator.init_weights)
    crit = crit.apply(Critic.init_weights)

    gen.train()
    crit.train()
    
    gen_optim = torch.optim.Adam(lr = config.g_learning_rate, 
                                 params = gen.parameters(),
                                 betas = (config.gan_betas[0], config.gan_betas[1]),
                                 eps=1e-08)
    
    crit_optim = torch.optim.Adam(lr = config.c_learning_rate, 
                                  params = crit.parameters(),
                                  betas = (config.gan_betas[0], config.gan_betas[1]),
                                  eps=1e-08) 
    
    c_loss_interval = []
    g_loss_interval= []
    c_loss_per_batch = []
    g_loss_per_batch = []
    acc_real_batch = []
    acc_fake_batch = []
    crit_sing0_first_layer = []
    crit_sing0_last_layer = []
    crit_sing1_first_layer = []
    crit_sing1_last_layer = []
    gen_sing0_first_layer = []
    gen_sing0_last_layer = []
    gen_sing1_first_layer = []
    gen_sing1_last_layer = []
    
    agnostic_idx = 0
    total_time = 0
    autoencoder_time = 0

    pretrain_crit = False
    print("pretraining critic: {}".format(pretrain_crit))
    if pretrain_crit == True:
        critic_pretrain_idx = 0
        critic_pretrain_n_times = 100
        pretrain_length = 100
    else:
        critic_pretrain_idx = 100
        critic_pretrain_n_times = 100
        pretrain_length = 100

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.gan_batch_size, all_data)):
            t0 = time.time()
            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            padded_batch = torch.LongTensor(padded_batch).to(config.device)
            crit_optim.zero_grad()
            t1 = time.time()
            with torch.no_grad():
                if autoencoder.name == "default_autoencoder":
                    z_real, _ = autoencoder.encoder(padded_batch, original_lens_batch)
                elif autoencoder.name == "cnn_autoencoder":
                    z_real, _ = autoencoder.encoder(padded_batch)
                elif autoencoder.name == "variational_autoencoder":
                    z_real = vae_encoding(autoencoder, padded_batch, original_lens_batch)
                else:
                    pass
            t2 = time.time()
            noise = sample_multivariate_gaussian(config)
            z_fake = gen(noise)
            real_score = crit(z_real)
            fake_score = crit(z_fake.detach())

            #grad_penalty = compute_grad_penalty(config, crit, z_real, z_fake)
            c_loss = - torch.mean(real_score) + torch.mean(fake_score)# + gp_lambda * grad_penalty

            c_loss_interval.append(c_loss.item())
            c_loss.backward()
            crit_optim.step()

            if (batch_idx % n_times_critic == 0 and critic_pretrain_idx >= pretrain_length) or (batch_idx % critic_pretrain_n_times == 0):
                critic_pretrain_idx += 1
                if unroll_steps > 0:
                    backup_crit = copy.deepcopy(crit)
                    for i in range(unroll_steps):
                        batch = sample_batch(config.gan_batch_size, all_data)
                        padded_batch = pad_batch(batch)
                        padded_batch = torch.LongTensor(padded_batch).to(config.device)
                        crit_optim.zero_grad()
                        with torch.no_grad():
                            if autoencoder.name == "default_autoencoder":
                                z_real, _ = autoencoder.encoder(padded_batch, original_lens_batch)
                            elif autoencoder.name == "cnn_autoencoder":
                                z_real, _ = autoencoder.encoder(padded_batch)
                            elif autoencoder.name == "variational_autoencoder":
                                z_real = vae_encoding(autoencoder, padded_batch, original_lens_batch)
                            else:
                                pass
                        real_score = crit(z_real)
                        noise = sample_multivariate_gaussian(config)
                        with torch.no_grad():
                            z_fake = gen(noise)
                        fake_score = crit(z_fake)
                        #grad_penalty = compute_grad_penalty(config, crit, z_real, z_fake)
                        c_loss = - torch.mean(real_score) + torch.mean(fake_score) #+ gp_lambda * grad_penalty
                        c_loss.backward()
                        crit_optim.step()
                    noise = sample_multivariate_gaussian(config)
                    gen_optim.zero_grad()
                    fake_score = crit(gen(noise))
                    if gdf == True:
                        gdf_loss = distribution_constraint(fitted_distribution, gen(noise), gdf_scaling_factor)
                        g_loss = - torch.mean(fake_score) + gdf_loss
                        if agnostic_idx % print_interval == 0:
                            print("gdf_loss", gdf_loss.item())
                    else:
                        g_loss = - torch.mean(fake_score)
                    g_loss.backward()
                    gen_optim.step()
                    g_loss_interval.append(g_loss.item())
                    crit.load(backup_crit)
                    del backup_crit
                else:
                    gen_optim.zero_grad()
                    fake_score = crit(gen(noise))
                    if gdf == True:
                        gdf_loss = distribution_constraint(fitted_distribution, gen(noise), gdf_scaling_factor)
                        g_loss = - torch.mean(fake_score) + gdf_loss
                        if agnostic_idx % print_interval == 0:
                            print("gdf_loss", gdf_loss.item())
                    else:
                        g_loss = - torch.mean(fake_score)
                    g_loss.backward()
                    gen_optim.step()
                    g_loss_interval.append(g_loss.item())

            t3 = time.time()

            if agnostic_idx > 0 and agnostic_idx % print_interval == 0:
                acc_real = torch.mean(real_score)
                acc_fake = torch.mean(fake_score)
                c_loss_per_batch.append(cutoff_scores(c_loss.item(), 500))
                g_loss_per_batch.append(cutoff_scores(g_loss.item(), 500))
                acc_real_batch.append(cutoff_scores(acc_real.item(), 500))
                acc_fake_batch.append(cutoff_scores(acc_fake.item(), 500))

                with torch.no_grad():
                    sing_vals = singular_values(gen, crit)
                    crit_sing0_first_layer.append(sing_vals[0])
                    crit_sing0_last_layer.append(sing_vals[1])
                    crit_sing1_first_layer.append(sing_vals[2])
                    crit_sing1_last_layer.append(sing_vals[3])
                    gen_sing0_first_layer.append(sing_vals[4])
                    gen_sing0_last_layer.append(sing_vals[5])
                    gen_sing1_first_layer.append(sing_vals[6])
                    gen_sing1_last_layer.append(sing_vals[7])

            if agnostic_idx > 0 and agnostic_idx % print_interval == 0:
                average_g_loss = sum(g_loss_interval) / len(g_loss_interval)
                average_c_loss = sum(c_loss_interval) / len(c_loss_interval)
                c_loss_interval = []
                g_loss_interval = []
                progress = ((batch_idx+1) * config.gan_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                progress = progress * 100
                progress = round(progress, 4)
                print("Progress {}% | Generator loss {:.6f}| Critic loss {:.6f}| Acc real {} | Acc fake {} over last {} batches"
                      .format(progress, average_g_loss, average_c_loss, acc_real.item(), acc_fake.item(), print_interval))
                
            agnostic_idx +=1
            total_time += t3 - t0
            autoencoder_time += t2 - t1

        if epoch_idx+1 >= 5 and (epoch_idx+1) % 2 == 0:
            save_gan(epoch_idx+1, autoencoder.name, gen, crit, norm_data)

    print("autoencoder as fraction of time", autoencoder_time / total_time)
    print("saving GAN...")
    save_gan(epoch_idx+1, autoencoder.name, gen, crit, norm_data)
    write_accs_to_file(acc_real_batch, acc_fake_batch, c_loss_per_batch, g_loss_per_batch, config.gan_batch_size, config.gan_betas[0], config.g_learning_rate)
    plot_gan_acc(acc_real_batch, acc_fake_batch, config.gan_batch_size, config.gan_betas[0], config.g_learning_rate, autoencoder.name)
    plot_gan_loss(c_loss_per_batch, g_loss_per_batch, config.gan_batch_size, config.gan_betas[0], config.g_learning_rate, autoencoder.name)
    plot_singular_values((crit_sing0_first_layer, 
                          crit_sing1_first_layer, 
                          crit_sing0_last_layer, 
                          crit_sing1_last_layer, 
                          gen_sing0_first_layer, 
                          gen_sing1_first_layer, 
                          gen_sing0_last_layer, 
                          gen_sing1_last_layer))