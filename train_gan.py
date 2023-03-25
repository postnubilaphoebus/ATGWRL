import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import os
import copy
import sys
from models import AutoEncoder, CNNAutoEncoder, Generator, Critic
from distribution_fitting import distribution_fitting, distribution_constraint
import random
import matplotlib.pyplot as plt
import time
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

def load_ae(config):
    print("Loading pretrained ae epoch 5...")
    model_5 = 'epoch_5_model_cnn_autoencoder.pth'
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_aes')
    model_5_path = os.path.join(saved_models_dir, model_5)
    model = CNNAutoEncoder(config)
    model.to(config.device)

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
              num_sents = 110_000,
              validation_size = 10_000,
              unroll_steps = 0,
              norm_data = True,
              gdf = False,
              gdf_scaling_factor = 0.001,
              num_epochs = 20,
              gp_lambda = 10,
              print_interval = 100,
              plotting_interval = 50_000, 
              n_times_critic = 1,
              data_path = "corpus_v20k_ids.txt", 
              vocab_path = "vocab_20k.txt"):
                  
    config.vocab_size = 20_000
    config.encoder_dim = 600
    config.word_embedding = 300
    print("num_epochs", num_epochs)
    autoencoder = load_ae(config)
    autoencoder.eval()

    data = load_data_from_file(data_path, num_sents)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))

    if norm_data == True:
        min_tensor, max_tensor = find_min_and_max(config, autoencoder, all_data)
    else:
        min_tensor, max_tensor = None, None

    if gdf == True:
        fitted_distribution = distribution_fitting(config, autoencoder, all_data, min_tensor, max_tensor)

    config.gan_batch_size = 256
    config.n_layers = 20

    print("batch size {}, n_layers {}, block_dim {}".format(config.gan_batch_size, config.n_layers, config.block_dim))
    print("n_times_critic", n_times_critic)

    crit_activation_function = "relu"
    gen_activation_function = "relu"

    print("activation G {}, activation C {}".format(gen_activation_function, crit_activation_function))
    gen = Generator(config.n_layers, config.block_dim, gen_activation_function).to(config.device)
    crit = Critic(config.n_layers, config.block_dim, crit_activation_function).to(config.device)
    #gen = load_gan(config)
    #crit = load_crit(config)

    print("unroll steps", unroll_steps)
    print("G lr", config.g_learning_rate)
    print("D lr", config.c_learning_rate)

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

    for epoch_idx in range(num_epochs):
        for batch_idx, batch in enumerate(yieldBatch(config.gan_batch_size, all_data)):
            t0 = time.time()
            original_lens_batch = real_lengths(batch)
            padded_batch = pad_batch(batch)
            padded_batch = torch.LongTensor(padded_batch).to(config.device)
            crit_optim.zero_grad()
            t1 = time.time()
            with torch.no_grad():
                z_real, _ = autoencoder.encoder(padded_batch)
            if norm_data:
                z_real = normalise(z_real, min_tensor, max_tensor)
            t2 = time.time()
            noise = sample_bernoulli(config)
            z_fake = gen(noise)
            real_score = crit(z_real)
            fake_score = crit(z_fake)

            grad_penalty = compute_grad_penalty(config, crit, z_real, z_fake)
            c_loss = - torch.mean(real_score) + torch.mean(fake_score) + gp_lambda * grad_penalty

            c_loss_interval.append(c_loss.item())
            c_loss.backward()
            crit_optim.step()

            if batch_idx % n_times_critic == 0:
                if unroll_steps > 0:
                    backup_crit = copy.deepcopy(crit)
                    for i in range(unroll_steps):
                        batch = sample_batch(config.gan_batch_size, all_data)
                        padded_batch = pad_batch(batch)
                        padded_batch = torch.LongTensor(padded_batch).to(config.device)
                        crit_optim.zero_grad()
                        with torch.no_grad():
                            z_real, _  = autoencoder.encoder(padded_batch, original_lens_batch)
                        if norm_data:
                            z_real = normalise(z_real, min_tensor, max_tensor)
                        real_score = crit(z_real)
                        noise = sample_bernoulli(config)
                        with torch.no_grad():
                            z_fake = gen(noise)
                        fake_score = crit(z_fake)
                        grad_penalty = compute_grad_penalty(config, crit, z_real, z_fake)
                        c_loss = - torch.mean(real_score) + torch.mean(fake_score) + gp_lambda * grad_penalty
                        c_loss.backward()
                        crit_optim.step()
                    noise = sample_bernoulli(config)
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
                c_loss_per_batch.append(cutoff_scores(c_loss.item()))
                g_loss_per_batch.append(cutoff_scores(g_loss.item()))
                acc_real_batch.append(cutoff_scores(acc_real.item()))
                acc_fake_batch.append(cutoff_scores(acc_fake.item()))

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

            #if agnostic_idx > 5000 and agnostic_idx % plotting_interval == 0:
                #plot_gan_acc(acc_real_batch, acc_fake_batch)
                #plot_gan_loss(c_loss_per_batch, g_loss_per_batch)

            if batch_idx > 0 and batch_idx % print_interval == 0:
                average_g_loss = sum(g_loss_interval) / len(g_loss_interval)
                average_c_loss = sum(c_loss_interval) / len(c_loss_interval)
                c_loss_interval = []
                g_loss_interval = []
                progress = ((batch_idx+1) * config.gan_batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                progress = progress * 100
                progress = round(progress, 4)
                print("Progress {}% | Generator loss {:.6f}| Critic loss {:.6f}| over last {} batches".format(progress, average_g_loss, average_c_loss, print_interval))
                
            agnostic_idx +=1
            total_time += t3 - t0
            autoencoder_time += t2 - t1

    print("autoencoder as fraction of time", autoencoder_time / total_time)
    print("saving GAN...")
    save_gan(epoch_idx+1, autoencoder.name, gen, crit)
    write_accs_to_file(acc_real_batch, acc_fake_batch, c_loss_per_batch, g_loss_per_batch, config.gan_batch_size, config.gan_betas[0], config.g_learning_rate)
    plot_gan_acc(acc_real_batch, acc_fake_batch, config.gan_batch_size, config.gan_betas[0], config.g_learning_rate)
    plot_gan_loss(c_loss_per_batch, g_loss_per_batch, config.gan_batch_size, config.gan_betas[0], config.g_learning_rate)
    plot_singular_values((crit_sing0_first_layer, 
                          crit_sing1_first_layer, 
                          crit_sing0_last_layer, 
                          crit_sing1_last_layer, 
                          gen_sing0_first_layer, 
                          gen_sing1_first_layer, 
                          gen_sing0_last_layer, 
                          gen_sing1_last_layer))
    #plot_grad_flow(crit.named_parameters())