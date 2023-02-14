import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import os
import sys
from ae import AutoEncoder
from gan import Generator, Critic
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
def save_gan(epoch, generator, critic):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_gan')
    if not os.path.exists(directory):
        os.makedirs(directory)

    critic_filename = 'critic_epoch_' + str(epoch) + '_model.pth'
    generator_filename = 'generator_epoch_' + str(epoch) + '_model.pth'
    critic_directory = os.path.join(directory, critic_filename)
    generator_directory = os.path.join(directory, generator_filename)
    torch.save(generator.state_dict(), generator_directory)
    torch.save(critic.state_dict(), critic_directory)

def sample_batch(data, batch_size):
    sample_num = random.randint(0, len(data) - batch_size - 1)
    data_batch = data[sample_num:sample_num+batch_size]
    return data_batch

def load_ae(config):
    print("Loading pretrained ae...")
    model_5 = 'epoch_5_model.pth'
    base_path = '/content/gdrive/MyDrive/ATGWRL/'
    saved_models_dir = os.path.join(base_path, r'saved_v´aes')
    model_5_path = os.path.join(saved_models_dir, model_5)
    model = AutoEncoder(config)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_5_path):
            model.load_state_dict(torch.load(model_5_path), strict = False)
        else:
            sys.exit("AE model path does not exist")
    else:
        sys.exit("AE path does not exist")

    return model

def compute_grad_penalty(config, critic, real_data, fake_data):
    B = real_data.size(0)
    alpha = torch.FloatTensor(np.random.random((B, 1)))
    alpha.to(config.device)
    sample = alpha * real_data + (1-alpha) * fake_data
    sample.requires_grad_(True)
    score = critic(sample)
    outputs = torch.FloatTensor(B, config.latent_dim).fill_(1.0)
    outputs.requires_grad_(False)
    outputs.to(config.device)
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
              validation_size,
              num_epochs = 15,
              gp_lambda = 10,
              print_interval = 100, 
              n_times_critic = 5,
              data_path = "corpus_v20k_ids.txt", 
              vocab_path = "vocab_20k.txt"):
    autoencoder = load_ae(config)
    autoencoder.eval()
    data = load_data_from_file(data_path)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))

    gen = Generator(config.n_layers, config.block_dim).to(config.device)
    crit = Critic(config.n_layers, config.block_dim).to(config.device)

    gen.train()
    crit.train()

    gen_optim = torch.optim.Adam(lr = config.gan_learning_rate, 
                                 params = gen.parameters(),
                                 betas = (0.9, 0.999),
                                 eps=1e-08)
    crit_optim = torch.optim.Adam(lr = config.gan_learning_rate, 
                                 params = crit.parameters(),
                                 betas = (0.9, 0.999),
                                 eps=1e-08)
    
    iterations = round(data_len / config.batch_size * num_epochs)
    c_loss_interval = []
    g_loss_interval= []

    epoch_equivalent = data_len // config.batch_size
    curr_epoch = 0

    for batch_idx in range(iterations):
        batch = sample_batch(config.batch_size, all_data)
        original_lens_batch = real_lengths(batch)
        padded_batch = pad_batch(batch)
        padded_batch = torch.LongTensor(padded_batch).to(config.device)

        with torch.no_grad():
            embedded = autoencoder.embedding_layer(padded_batch)
            z_real = autoencoder.encoder(embedded, original_lens_batch)

        crit_optim.zero_grad()
        noise = torch.from_numpy(np.random.normal(0, 1, (config.batch_size, config.latent_dim))).float()
        noise = noise.to(config.device)
        z_fake = gen(noise)
        real_score = crit(z_real)
        fake_score = crit(z_fake)

        grad_penalty = compute_grad_penalty(config, crit, z_real, z_fake)
        c_loss = - torch.mean(real_score) + torch.mean(fake_score) + gp_lambda * grad_penalty

        c_loss_interval.append(c_loss.item())

        c_loss.backward()
        crit_optim.step()

        if batch_idx % n_times_critic == 0:
            gen_optim.zero_grad()
            fake_score = crit(gen(noise))
            g_loss = - torch.mean(fake_score)
            g_loss.backward()
            gen_optim.step()
            g_loss_interval.append(g_loss.item())

        if batch_idx > 0 and batch_idx % print_interval == 0:
            average_g_loss = sum(g_loss_interval) / len(g_loss_interval)
            average_c_loss = sum(c_loss_interval) / len(c_loss_interval)
            c_loss_interval = []
            g_loss_interval= []
            progress = batch_idx / iterations * 100
            print("Batches complete {}| Progress {}%".format(batch_idx, progress))
            print("Generator loss {:.6f}| Critic loss {:.6f}| over last {} batches".format(average_g_loss, average_c_loss, print_interval))

        if batch_idx % epoch_equivalent == 0:
            curr_epoch += 1
            print("Checkpoint, saving generator and critic")
            save_gan(curr_epoch, gen, crit)

    print("GAN training done, saving models")
    save_gan(num_epochs, gen, crit)