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
import random
import matplotlib.pyplot as plt
import time
from matplotlib.lines import Line2D
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

def cutoff_scores(score, cutoff_val = 20):
    if score > cutoff_val:
        return cutoff_val
    elif score < -cutoff_val:
        return - cutoff_val
    else:
        return score
    
def sample_multivariate_gaussian(config):
    mean = np.zeros(shape=config.latent_dim) 
    cov = np.random.normal(0, 1, size=(config.latent_dim, config.latent_dim))
    cov = np.dot(cov, cov.transpose())
    cov_min = -np.amin(cov)
    cov_max = np.amax(cov)
    matrix_max = max(float(cov_min), float(cov_max))
    cov = cov / matrix_max
    noise = torch.from_numpy(np.random.default_rng().multivariate_normal(mean, cov, (config.gan_batch_size))).float()
    noise = noise.to(config.device)
    return noise
    
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'gradient_flow')
    if not os.path.exists(directory):
        os.makedirs(directory)
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().detach())
            max_grads.append(p.grad.abs().max().cpu().detach())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    final_directory = os.path.join(directory, "gradient_flow_end")
    plt.savefig(final_directory, dpi=300)
    plt.close()

def plot_gan_acc(real_score, fake_score, batch_size, moment, rate):
    epochs = len(real_score)
    real_score = np.array(real_score)
    fake_score = np.array(fake_score)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_gan_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'Plotted accs after ' + str(epochs) + 'batches (in 100s).png' + ' bs' + str(batch_size) + ' mom' + str(moment) + ' learning rate ' + str(rate) + '.png'
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
    plt.grid(True)
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

def plot_gan_loss(c_loss, g_loss, batch_size, moment, rate):
    epochs = len(c_loss)
    c_loss = np.array(c_loss)
    g_loss = np.array(g_loss)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_gan_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'Plotted loss after ' + str(epochs) + 'batches (in 100s)' + ' bs' + str(batch_size) + ' mom' + str(moment) + ' learning rate ' + str(rate) + '.png'
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
    plt.grid(True)
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

def plot_singular_values(sing_val_list):
    c00 = np.array(sing_val_list[0])
    c01 = np.array(sing_val_list[1])
    c10 = np.array(sing_val_list[2])
    c11 = np.array(sing_val_list[3])
    g00 = np.array(sing_val_list[4])
    g01 = np.array(sing_val_list[5])
    g10 = np.array(sing_val_list[6])
    g11 = np.array(sing_val_list[7])
    temp = len(c00)
    epochs = []
    for i in range(temp):
        epochs.append(i)
    epochs = np.array(epochs)
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_singular_values')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.plot(epochs, c00, label = "c0_first_layer")
    plt.plot(epochs, c01, label = "c1_first_layer")
    plt.xlabel('Batches (in 100s)')
    plt.ylabel('Singular value')
    plt.title("Singular values for first layer")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(directory, "critic_first_layer"), dpi=300)
    plt.close()

    plt.plot(epochs, c10, label = "c0_last_layer")
    plt.plot(epochs, c11, label = "c1_last_layer")
    plt.xlabel('Batches (in 100s)')
    plt.ylabel('Singular value')
    plt.title("Singular values for last layer")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(directory, "critic_last_layer"), dpi=300)
    plt.close()

    plt.plot(epochs, g00, label = "g0_first_layer")
    plt.plot(epochs, g01, label = "g1_first_layer")
    plt.xlabel('Batches (in 100s)')
    plt.ylabel('Singular value')
    plt.title("Singular values for first layer")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(directory, "generator_first_layer"), dpi=300)
    plt.close()

    plt.plot(epochs, g10, label = "g0_last_layer")
    plt.plot(epochs, g11, label = "g1_last_layer")
    plt.xlabel('Batches (in 100s)')
    plt.ylabel('Singular value')
    plt.title("Singular values for last layer")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(directory, "generator_last_layer"), dpi=300)
    plt.close()

def write_accs_to_file(acc_real, acc_fake, c_loss, g_loss, batch_size, fam, lr):
    with open("gan_results.txt", "a") as f:
        idx = 0
        ar = "acc_real "
        af = "acc_fake "
        cl = "c_loss "
        gl = "g_loss "
        
        f.write("\n")
        f.write("##################################################################################################################################" + "\n")
        f.write("batch size " + str(batch_size) + " " + "first Adam moment " + str(fam) + " learning rate " + str(lr) + "\n")
        f.write("##################################################################################################################################" + "\n")

        f.write("Final accs " + ar + str(acc_real[-1]) + " " + af + str(acc_fake[-1]) + " " + cl + str(c_loss[-1]) + " " + gl + str(g_loss[-1]) + "\n")
        f.write("##################################################################################################################################" + "\n" + "\n")
        for a_r, a_f, c_l, c_g in zip(acc_real, acc_fake, c_loss, g_loss):
            f.write("batch_id(100s) " + str(idx) + " " + ar + str(a_r) + " " + af + str(a_f) + " " + cl + str(c_l) + " " + gl + str(c_g) + "\n")
            idx+=1

    f.close()

def sample_batch(batch_size, data):
    sample_num = random.randint(0, len(data) - batch_size - 1)
    data_batch = data[sample_num:sample_num+batch_size]
    return data_batch

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

def singular_values(gen, crit):
    # singular values for critic

    first_layer, _ = torch.topk(torch.linalg.svdvals(crit.net[0].net[0].weight), k = 2)
    c00 = first_layer[0].item()
    c01 = first_layer[1].item()
    last_layer, _ = torch.topk(torch.linalg.svdvals(crit.net[-1].net[-1].weight), k = 2)
    c10 = last_layer[0].item()
    c11 = last_layer[1].item()

    # singular values for generator

    first_layer, _ = torch.topk(torch.linalg.svdvals(gen.net[0].net[0].weight), k = 2)
    g00 = first_layer[0].item()
    g01 = first_layer[1].item()
    last_layer, _ = torch.topk(torch.linalg.svdvals(gen.net[-1].net[-1].weight), k = 2)
    g10 = last_layer[0].item()
    g11 = last_layer[1].item()

    return (c00, c01, c10, c11, g00, g01, g10, g11)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def train_gan(config, 
              validation_size = 10_000,
              unroll_steps = 0,
              num_epochs = 30,
              gp_lambda = 10,
              print_interval = 100,
              plotting_interval = 50_000, 
              n_times_critic = 1,
              data_path = "corpus_v20k_ids.txt", 
              vocab_path = "vocab_20k.txt"):
                  
    config.vocab_size = 20_000
    #print("plotting_interval", plotting_interval)
    config.encoder_dim = 600
    config.word_embedding = 300
    print("num_epochs", num_epochs)
    autoencoder = load_ae(config)
    autoencoder.eval()
    data = load_data_from_file(data_path, 110_000)
    val, all_data = data[:validation_size], data[validation_size:]
    #all_data = data
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))

    config.gan_batch_size = 128
    config.n_layers = 10

    print("batch size {}, n_layers {}, block_dim {}".format(config.gan_batch_size, config.n_layers, config.block_dim))
    print("n_times_critic", n_times_critic)

    crit_activation_function = "relu"
    gen_activation_function = "relu"

    print("activation G {}, activation C {}".format(gen_activation_function, crit_activation_function))
    gen = Generator(config.n_layers, config.block_dim, gen_activation_function).to(config.device)
    crit = Critic(config.n_layers, config.block_dim, crit_activation_function).to(config.device)
    #gen = load_gan(config)
    #crit = load_crit(config)

    config.g_learning_rate = 1e-4
    config.c_learning_rate = 1e-4

    print("unroll steps", unroll_steps)
    print("G lr", config.g_learning_rate)
    print("D lr", config.c_learning_rate)

    gen = gen.apply(Generator.init_weights)
    crit = crit.apply(Critic.init_weights)

    gen.train()
    crit.train()

    # use binomial noise

    gen_optim = torch.optim.Adam(lr = config.g_learning_rate, 
                                 params = gen.parameters(),
                                 betas = (0.5, 0.9),
                                 eps=1e-08)
    
    crit_optim = torch.optim.Adam(lr = config.c_learning_rate, 
                                  params = crit.parameters(),
                                  betas = (0.5, 0.9),
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
            t2 = time.time()
            noise = sample_multivariate_gaussian(config)
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
                        real_score = crit(z_real)
                        noise = sample_multivariate_gaussian(config)
                        with torch.no_grad():
                            z_fake = gen(noise)
                        fake_score = crit(z_fake)
                        grad_penalty = compute_grad_penalty(config, crit, z_real, z_fake)
                        c_loss = - torch.mean(real_score) + torch.mean(fake_score) + gp_lambda * grad_penalty
                        c_loss.backward()
                        crit_optim.step()
                    noise = torch.from_numpy(np.random.normal(0, 1, (config.gan_batch_size, config.latent_dim))).float()
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

        #print("Checkpoint, saving generator and critic")
        #save_gan(epoch_idx+1, gen, crit)

    print("autoencoder as fraction of time", autoencoder_time / total_time)
    print("saving GAN...")
    save_gan(epoch_idx+1, gen, crit)
    write_accs_to_file(acc_real_batch, acc_fake_batch, c_loss_per_batch, g_loss_per_batch, config.gan_batch_size, 0.5, config.g_learning_rate)
    plot_gan_acc(acc_real_batch, acc_fake_batch, config.gan_batch_size, 0.5, config.g_learning_rate)
    plot_gan_loss(c_loss_per_batch, g_loss_per_batch, config.gan_batch_size, 0.5, config.g_learning_rate)
    plot_singular_values((crit_sing0_first_layer, 
                          crit_sing1_first_layer, 
                          crit_sing0_last_layer, 
                          crit_sing1_last_layer, 
                          gen_sing0_first_layer, 
                          gen_sing1_first_layer, 
                          gen_sing0_last_layer, 
                          gen_sing1_last_layer))
    #plot_grad_flow(crit.named_parameters())

