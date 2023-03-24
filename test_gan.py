import torch
import numpy as np
import os
import sys
from models import AutoEncoder
from models import Generator
from utils.helper_functions import load_vocab

def load_ae(config):
    print("Loading pretrained ae...")
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
    model_15 = 'generator_epoch_15_model.pth'
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_gan')
    model_15_path = os.path.join(saved_models_dir, model_15)
    model = Generator(config)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_15_path):
            model.load_state_dict(torch.load(model_15_path, map_location=torch.device(config.device)), strict = False)
        else:
            sys.exit("GAN model path does not exist")
    else:
        sys.exit("GAN path does not exist")

    return model

def test(config, vocab_path = "vocab_20k.txt"):
    autoencoder = load_ae(config)
    generator = load_gan(config)
    autoencoder.eval()
    generator.eval()
    vocab, revvocab = load_vocab(vocab_path)
    noise = torch.from_numpy(np.random.normal(0, 1, (100, config.latent_dim))).float()
    noise = noise.to(config.device)
    with torch.no_grad():
        z_fake = generator(noise)
        decoded = autoencoder.decoder(z_fake)
        logits = autoencoder.hidden_to_vocab(decoded)
        sentences = torch.argmax(logits, dim = -1)
    for sentence in sentences:
        string_sent = []
        for word in sentence:
            if revvocab[word] not in [0,1]:
                string_sent.append(revvocab[word])
            else:
                break
        to_print = ' '.join(string_sent)
        print(to_print)

    print("Demonstration done")

            
                



            
        