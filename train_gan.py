import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import os
import copy
import sys
from models import AutoEncoder, Generator, Critic
import random
import matplotlib.pyplot as plt
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
                                   
def save_gan(epoch, generator, critic, batch_size, moment):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_gan')
    if not os.path.exists(directory):
        os.makedirs(directory)

    critic_filename = 'critic_epoch_' + str(epoch) + '_model_' + str(batch_size) + '_moment_' + str(moment) + '_.pth'
    generator_filename = 'generator_epoch_' + str(epoch) + '_model_' + str(batch_size) + '_moment_' + str(moment) + '_.pth'
    critic_directory = os.path.join(directory, critic_filename)
    generator_directory = os.path.join(directory, generator_filename)
    torch.save(generator.state_dict(), generator_directory)
    torch.save(critic.state_dict(), critic_directory)

def sample_batch(batch_size, data):
    sample_num = random.randint(0, len(data) - batch_size - 1)
    data_batch = data[sample_num:sample_num+batch_size]
    return data_batch

def load_ae(config):
    print("Loading pretrained ae epoch 5...")
    model_5 = 'epoch_5_model.pth'
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_aes')
    model_5_path = os.path.join(saved_models_dir, model_5)
    model = AutoEncoder(config)
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

def plot_gan_acc(real_score, fake_score, batch_size, moment):
    epochs = len(real_score)
    real_score = np.array(real_score)
    fake_score = np.array(fake_score)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_gan_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'Plotted accs after ' + str(epochs) + 'batches (in 100s).png' + ' bs' + str(batch_size) + ' mom' + str(moment) + '.png'
    final_directory = os.path.join(directory, filename)
    temp = epochs
    epochs = []
    for i in range(temp):
        epochs.append(i)
    epochs = np.array(epochs)
    plt.plot(epochs, real_score, label = 'score_real')
    plt.plot(epochs, fake_score, label = 'score_fake')
    plt.xlabel('Batches (in 100s)')
    plt.ylabel('Critic score')
    plt.title('Critic scores plotted over ' + str(temp) + ' batches (in 100s)' + ' bs' + str(batch_size) + ' mom' + str(moment), fontsize = 10)
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

def plot_gan_loss(c_loss, g_loss, batch_size, moment):
    epochs = len(c_loss)
    c_loss = np.array(c_loss)
    g_loss = np.array(g_loss)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_gan_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'Plotted loss after ' + str(epochs) + 'batches (in 100s)' + ' bs' + str(batch_size) + ' mom' + str(moment) + '.png'
    final_directory = os.path.join(directory, filename)
    temp = epochs
    epochs = []
    for i in range(temp):
        epochs.append(i)
    epochs = np.array(epochs)
    plt.plot(epochs, c_loss, label = 'critic loss')
    plt.plot(epochs, g_loss, label = 'generator loss')
    #plt.yscale('log')
    plt.xlabel('Batches (in 100s)')
    plt.ylabel('Loss')
    
    plt.title('G and C losses plotted over ' + str(temp) + ' batches (in 100s)' + ' bs' + str(batch_size) + ' mom' + str(moment), fontsize = 10)
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()


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

def write_accs_to_file(acc_real, acc_fake, c_loss, g_loss, batch_size, fam):
    with open("gan_results.txt", "a") as f:
        idx = 0
        ar = "acc_real "
        af = "acc_fake "
        cl = "c_loss "
        gl = "g_loss "
        f.write("\n")
        f.write("##################################################################################################################################" + "\n")
        f.write("batch size " + str(batch_size) + " " + "first Adam moment " + str(fam) + "\n")
        f.write("##################################################################################################################################" + "\n")
        f.write("Final accs " + ar + str(acc_real[-1]) + " " + af + str(acc_fake[-1]) + " " + cl + str(c_loss[-1]) + " " + gl + str(g_loss[-1]) + "\n")
        f.write("##################################################################################################################################" + "\n" + "\n")
        for a_r, a_f, c_l, c_g in zip(acc_real, acc_fake, c_loss, g_loss):
            f.write("batch_id(100s) " + str(idx) + " " + ar + str(a_r) + " " + af + str(a_f) + " " + cl + str(c_l) + " " + gl + str(c_g) + "\n")
            idx+=1

    f.close()

def train_gan(config, 
              validation_size = 10_000,
              unroll_steps = 0,
              num_epochs = 15,
              gp_lambda = 10,
              print_interval = 100,
              plotting_interval = 50_000, 
              n_times_critic = 5,
              data_path = "corpus_v20k_ids.txt", 
              vocab_path = "vocab_20k.txt"):
                  
    config.vocab_size = 20_000
    print("plotting_interval", plotting_interval)
    print("num_epochs", num_epochs)
    autoencoder = load_ae(config)
    autoencoder.eval()
    data = load_data_from_file(data_path, 200_000)
    #val, all_data = data[:validation_size], data[validation_size:]
    all_data = data
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))

    batch_sizes = [16, 32, 50, 128]
    momentum = [0.5, 0]

    for batch_size in batch_sizes:
        for moment in momentum:
            c_loss_interval = []
            g_loss_interval= []
            c_loss_per_batch = []
            g_loss_per_batch = []
            acc_real_batch = []
            acc_fake_batch = []
            
            agnostic_idx = 0
            config.batch_size = batch_size

            gen = Generator(config.n_layers, config.block_dim).to(config.device)
            crit = Critic(config.n_layers, config.block_dim).to(config.device)
            #gen = load_gan(config)
            #crit = load_crit(config)
            gen = gen.apply(Generator.init_weights)
            crit = crit.apply(Critic.init_weights)

            gen.train()
            crit.train()

            gen_optim = torch.optim.Adam(lr = config.gan_learning_rate, 
                                        params = gen.parameters(),
                                        betas = (moment, 0.9),
                                        eps=1e-08)
            crit_optim = torch.optim.Adam(lr = config.gan_learning_rate, 
                                        params = crit.parameters(),
                                        betas = (moment, 0.9),
                                        eps=1e-08) 

            for epoch_idx in range(num_epochs):
                for batch_idx, batch in enumerate(yieldBatch(config.batch_size, all_data)):
                    original_lens_batch = real_lengths(batch)
                    padded_batch = pad_batch(batch)
                    padded_batch = torch.LongTensor(padded_batch).to(config.device)

                    crit_optim.zero_grad()
                    with torch.no_grad():
                        z_real, _ = autoencoder.encoder(padded_batch, original_lens_batch)
                        
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
                        if unroll_steps > 0:
                            backup_crit = copy.deepcopy(crit)
                            for i in range(unroll_steps):
                                batch = sample_batch(config.batch_size, all_data)
                                padded_batch = pad_batch(batch)
                                padded_batch = torch.LongTensor(padded_batch).to(config.device)
                                crit_optim.zero_grad()
                                with torch.no_grad():
                                    z_real, _  = autoencoder.encoder(padded_batch, original_lens_batch)
                                real_score = crit(z_real)
                                noise = torch.from_numpy(np.random.normal(0, 1, (config.batch_size, config.latent_dim))).float()
                                noise = noise.to(config.device)
                                with torch.no_grad():
                                    z_fake = gen(noise)
                                fake_score = crit(z_fake)
                                grad_penalty = compute_grad_penalty(config, crit, z_real, z_fake)
                                c_loss = - torch.mean(real_score) + torch.mean(fake_score) + gp_lambda * grad_penalty
                                c_loss.backward()
                                crit_optim.step()
                            noise = torch.from_numpy(np.random.normal(0, 1, (config.batch_size, config.latent_dim))).float()
                            noise = noise.to(config.device)
                            gen_optim.zero_grad()
                            fake_score = crit(gen(noise))
                            g_loss = - torch.mean(fake_score)
                            g_loss.backward()
                            gen_optim.step()
                            g_loss_interval.append(g_loss.item())
                            crit.load(backup_crit)
                            del backup_crit
                        else:
                            gen_optim.zero_grad()
                            fake_score = crit(gen(noise))
                            g_loss = - torch.mean(fake_score)
                            g_loss.backward()
                            gen_optim.step()
                            g_loss_interval.append(g_loss.item())

                    if agnostic_idx > 1000 and agnostic_idx % print_interval == 0:
                        acc_real = torch.mean(real_score)
                        acc_fake = torch.mean(fake_score)
                        c_loss_per_batch.append(c_loss.item())
                        g_loss_per_batch.append(g_loss.item())
                        acc_real_batch.append(acc_real.item())
                        acc_fake_batch.append(acc_fake.item())

                    #if agnostic_idx > 1000 and agnostic_idx % plotting_interval == 0:
                        #plot_gan_acc(acc_real_batch, acc_fake_batch)
                        #plot_gan_loss(c_loss_per_batch, g_loss_per_batch)

                    if batch_idx > 0 and batch_idx % print_interval == 0:
                        average_g_loss = sum(g_loss_interval) / len(g_loss_interval)
                        average_c_loss = sum(c_loss_interval) / len(c_loss_interval)
                        c_loss_interval = []
                        g_loss_interval = []
                        progress = ((batch_idx+1) * config.batch_size / data_len / num_epochs) + (epoch_idx / num_epochs)
                        progress = progress * 100
                        print("Batches complete {}| Progress {}%".format(batch_idx, progress))
                        print("Generator loss {:.6f}| Critic loss {:.6f}| over last {} batches".format(average_g_loss, average_c_loss, print_interval))
                        
                    agnostic_idx +=1

                #print("Checkpoint, saving generator and critic")
                #save_gan(epoch_idx+1, gen, crit)

            print("writing accs to file...")
            write_accs_to_file(acc_real_batch, acc_fake_batch, c_loss_per_batch, g_loss_per_batch, batch_size, moment)
            print("plotting...")
            plot_gan_acc(acc_real_batch, acc_fake_batch, batch_size, moment)
            plot_gan_loss(c_loss_per_batch, g_loss_per_batch, batch_size, moment)
            print("saving...")
            save_gan(num_epochs, gen, crit, batch_size, moment)

    print("n_times_critic", n_times_critic)
    print("batch size {}, learning rate {}, n_layers {}, block_dim {}".format(config.batch_size, config.gan_learning_rate, config.n_layers, config.block_dim))

