import torch
import numpy as np
import os
import sys
import warnings
from models import AutoEncoder, CNNAutoEncoder, VariationalAutoEncoder, Generator
from utils.helper_functions import load_vocab, sample_multivariate_gaussian, re_scale, rouge_and_bleu
from rouge_score import rouge_scorer
#from BARTScore.bart_score import BARTScorer
import language_tool_python
from collections import defaultdict
from itertools import groupby

def load_ae(model_name, config):
    weights_matrix = None

    if model_name == "default_autoencoder":
        model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)
        #model_5 = "epoch_5_model_default_autoencoder.pth"
        model_5 = "epoch_10_model_default_autoencoder_regime_normal_latent_mode_dropout.pth"
    elif model_name == "cnn_autoencoder":
        model = CNNAutoEncoder(config, weights_matrix)
        model = model.apply(CNNAutoEncoder.init_weights)
        model.to(model.device)
        model_5 = 'epoch_7_model_cnn_autoencoder_regime_normal_latent_mode_dropout.pth'
        #model_5 = 'epoch_5_model_cnn_autoencoder.pth'
    elif model_name == "variational_autoencoder":
        model = VariationalAutoEncoder(config, weights_matrix)
        model = model.apply(VariationalAutoEncoder.init_weights)
        model.to(model.device)
        model_5 = "epoch_5_model_variational_autoencoder.pth"
    else:
        warnings.warn("Provided invalid model name. Loading default autoencoder...")
        model = AutoEncoder(config, weights_matrix)
        model = model.apply(AutoEncoder.init_weights)
        model.to(model.device)
        model_5 = "epoch_5_model_default_autoencoder.pth"

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

def load_gan(config, filename):
    print("Loading pretrained generator...")
    base_path = os.getcwd()
    saved_models_dir = os.path.join(base_path, r'saved_gan')
    model_15_path = os.path.join(saved_models_dir, filename)
    model = Generator(n_layers = 10, block_dim = 100)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_15_path):
            model.load_state_dict(torch.load(model_15_path, map_location=torch.device(config.device)), strict = False)
        else:
            sys.exit("GAN model path does not exist")
    else:
        sys.exit("GAN path does not exist")

    return model

def test(config, vocab_path = "vocab_40k.txt"):
    #gan_filename = "generator_epoch_150_default_autoencoder_model.pth"
    #gan_filename = "generator_epoch_170_cnn_autoencoder_model.pth"
    #gan_filename = "generator_epoch_170_default_autoencoder_model.pth"
    gan_filename = "generator_epoch_180_default_autoencoder_model.pth"
    config.gan_batch_size = 64

    ##max_tensor_name = "maximum_for_rescaling_default_autoencoder_.pth"
    #min_tensor_name = "minimum_for_rescaling_default_autoencoder_.pth"
    # cnn_autoencoder, default_autoencoder, variational_autoencoder
    autoencoder = load_ae("default_autoencoder", config)
    generator = load_gan(config, gan_filename)
    autoencoder.eval()
    generator.eval()
    vocab, revvocab = load_vocab(vocab_path, 40_001)
    noise = sample_multivariate_gaussian(config)
    #base_path = os.getcwd()
    #saved_models_dir = os.path.join(base_path, r'saved_gan')
    #tensor_path_max = os.path.join(saved_models_dir, max_tensor_name)
    #tensor_path_min = os.path.join(saved_models_dir, min_tensor_name)
    #max_tensor = torch.load(tensor_path_max, map_location=torch.device(config.device))
    #min_tensor = torch.load(tensor_path_min, map_location=torch.device(config.device))

    tool = language_tool_python.LanguageTool('en-US') 
    british_english_mistake = "is British English"

    with torch.no_grad():
        z_fake = generator(noise)
        decoded = autoencoder.decoder(z_fake, None, None)
        logits = autoencoder.hidden_to_vocab(decoded)
        sentences = torch.argmax(logits, dim = -1)
    sentences = torch.transpose(sentences, 1, 0)
    sentences = sentences.cpu().detach().tolist()

    scorer = rouge_scorer.RougeScorer(['rouge1', "rouge2", "rouge3", 'rouge4'], use_stemmer=True)
    scores = [0] * 8

    comparisons = 0
    print("calculate self-bleu and self-rouge scores...")

    for idx, sentence_x in enumerate(sentences):
        for jdx, sentence_y in enumerate(sentences):
            if idx >= jdx:
                continue
            else:
                string_sent_x = []
                string_sent_y = []

                for word in sentence_x:
                    if word not in [0,1]:
                        string_sent_x.append(revvocab[word])
                    else:
                        break
                    last_word = word
                to_print_x = ' '.join(string_sent_x)
                for word in sentence_y:
                    if word not in [0,1]:
                        string_sent_y.append(revvocab[word])
                    else:
                        break
                    last_word = word
                to_print_y = ' '.join(string_sent_y)
                interim = rouge_and_bleu(to_print_x, to_print_y, scorer)
                comparisons += 1
                for i in range(len(interim)):
                    scores[i] += interim[i]

    scores = [x / comparisons for x in scores]
    score_names = ["rouge1", "rouge2", "rouge3", "rouge4", "bleu1", "bleu2", "bleu3", "bleu4"]
    
    for score, name in zip(scores, score_names):
        print("{}: {}".format(name, score))

    print("calculate grammatical errors and word repetition...")
    errors = 0
    word_reps = 0
    for sentence in sentences:
        string_sent = []
        last_word = -1
        curr_counted = 0
        for word in sentence:
            if word == last_word:
                curr_counted += 1
            if word not in [0,1]:
                string_sent.append(revvocab[word])
            else:
                break
            last_word = word
            

        if curr_counted > 0:
            curr_counted += 1
        to_print = ' '.join(string_sent)
        matches = tool.check(to_print)
        error_sent = len([rule for rule in matches if (rule.category != 'CASING' and british_english_mistake not in rule.message)])
        sent_len = to_print.count(' ') + 1
        errors += (error_sent / sent_len)
        word_reps += (curr_counted / sent_len)
        print(to_print + "\n")

    print("average errors per sentence, normalised", errors / len(sentences))
    print("subsequently repeated words per sentence, normalised", word_reps / len(sentences))
    print("Demonstration done")

            
                



            
        