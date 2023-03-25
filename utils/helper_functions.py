from datasets import load_dataset
from nltk.tokenize import WhitespaceTokenizer, TreebankWordTokenizer
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_scoring
from nltk.translate.bleu_score import sentence_bleu
from matplotlib.lines import Line2D
import os.path
from tqdm import tqdm
import sys
import regex as re
import numpy as np
import math
import time
import torch
import truecase
import random
import spacy
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFD, StripAccents
from tokenizers import normalizers
from sklearn.metrics.pairwise import cosine_similarity

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3

MAX_SENT_LEN = 28 # 20 (+ 8 for punctuation)
MAX_TOKEN_LENGTH = 28

common_nes ={"PERSON": "PERSON_token", "ORG": "ORG_token", "GPE": "GPE_token"}

def inverse_sigmoid_schedule(iterations, sigmoid_rate):
    a = sigmoid_rate/(sigmoid_rate + math.exp(iterations/sigmoid_rate))
    return a

def autoencoder_info(model, config):
    print("vocab size:", config.vocab_size)
    print("batch_size:", config.ae_batch_size)
    print("latent dimension:", config.latent_dim)
    print("layer normalisation:", config.layer_norm)
    print("word embedding size:", config.word_embedding)
    print("using pretrained embedding:", config.pretrained_embedding)
    print("dropout probability:", config.dropout_prob)
    print("learning rate:", config.ae_learning_rate)
    print("ae betas:", *config.ae_betas)
    if model.name == "default_autoencoder":
        print("bidirectional:", config.bidirectional)
        print("attention:", config.attn_bool)
        if config.attn_bool:
            print("attention heads:", config.num_attn_heads)
        print("encoder dimension:", config.encoder_dim)
        print("decoder dimension:", config.decoder_dim)
    elif model.name == "cnn_autoencoder":
        print("max_pool_kernel:", config.max_pool_kernel)
        print("kernel_sizes:", *config.kernel_sizes)
        print("out channels:", config.out_channels)
    else:
        pass

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

def save_gan(epoch, autoencoder_name, generator, critic):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_gan')
    if not os.path.exists(directory):
        os.makedirs(directory)

    critic_filename = 'critic_epoch_' + str(epoch) + "_" + autoencoder_name + '_model.pth'
    generator_filename = 'generator_epoch_' + str(epoch) + "_" + autoencoder_name + '_model.pth'
    critic_directory = os.path.join(directory, critic_filename)
    generator_directory = os.path.join(directory, generator_filename)
    torch.save(generator.state_dict(), generator_directory)
    torch.save(critic.state_dict(), critic_directory)

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

def cutoff_scores(score, cutoff_val = 5):
    if score > cutoff_val:
        return cutoff_val
    elif score < -cutoff_val:
        return - cutoff_val
    else:
        return score
    
def find_min_and_max(config,
                      model,
                      data):
    # find min and max of data to normalise data
    # between 0 and 1
    print("finding minimum and maximum in data...")

    maximum = torch.full((config.latent_dim,), -1_000_000.0, dtype=torch.float64)
    minimum = torch.full((config.latent_dim,), 1_000_000.0, dtype=torch.float64)

    for batch_idx, batch in enumerate(yieldBatch(1_000, data)):
        original_lens_batch = real_lengths(batch)
        padded_batch = pad_batch(batch)
        padded_batch = torch.LongTensor(padded_batch).to(model.device)
        with torch.no_grad():
            if model.name == "variational_autoencoder":
                output = model.encoder(padded_batch, original_lens_batch)
                # extract last hidden state
                context = []
                for sequence, unpadded_len in zip(output, original_lens_batch):
                    context.append(sequence[unpadded_len-1, :])
                context = torch.stack((context))
                z = model.reparameterize(model.z_mean(context), model.z_log_var(context))
            elif model.name == "cnn_autoencoder":
                z, _ = model.encoder(padded_batch)
            else:
                z, _ = model.encoder(padded_batch, original_lens_batch)
            temp_max, _ = torch.max(z, dim = 0)
            temp_min, _ = torch.min(z, dim = 0)
            maximum = torch.maximum(maximum, temp_max)
            minimum = torch.minimum(minimum, temp_min)

    return minimum, maximum

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

def normalise(z, min, max):
    max_diff = max - min
    z = [(x - min) / (max_diff) for x in z]
    z = torch.stack((z)).float()
    return z

def re_scale(z, min, max):
    max_diff = max - min
    z = [(max - x) / (max_diff) for x in z]
    z = torch.stack((z)).float()
    return z

def sample_bernoulli(config):
    probs = torch.rand(size=(config.gan_batch_size, config.latent_dim))
    bernoulli = torch.bernoulli(probs).to(config.device)
    return bernoulli

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

def return_bert_score(pred, target, device, batch_size):
    bs = torch.mean(bert_scoring(pred, target, lang="en", device = device, batch_size=batch_size)[-1]).item()
    return bs

def rouge_and_bleu(pred, target, rouge_scorer, verbose = False):
    smoothie = SmoothingFunction().method4
    rouge_score = rouge_scorer.score(target, pred)
    r1_fscore = round(rouge_score["rouge1"].fmeasure, 4)
    r2_fscore= round(rouge_score["rouge2"].fmeasure, 4)
    r3_fscore= round(rouge_score["rouge3"].fmeasure, 4)
    r4_fscore = round(rouge_score["rouge4"].fmeasure, 4)
    bleu1 = round(sentence_bleu(target, pred, smoothing_function=smoothie, weights=(1, 0, 0, 0)), 4)
    bleu2 = round(sentence_bleu(target, pred, smoothing_function=smoothie, weights=(0, 1, 0, 0)), 4)
    bleu3 = round(sentence_bleu(target, pred, smoothing_function=smoothie, weights=(0, 0, 1, 0)), 4)
    bleu4 = round(sentence_bleu(target, pred, smoothing_function=smoothie), 4)
    if verbose:
        print("bleu1", bleu1)
        print("bleu2", bleu2)
        print("bleu3", bleu3)
        print("bleu4", bleu4)
        print("rouge1", r1_fscore)
        print("rouge2", r2_fscore)
        print("rouge3", r3_fscore)
        print("rouge4", r4_fscore)
    return (r1_fscore, r2_fscore, r3_fscore, r4_fscore, \
           bleu1, bleu2, bleu3, bleu4)

def matrix_from_pretrained_embedding(vocab, vocab_size, emb_dim, pretrained_embed):
    weights_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, emb_dim))
    words_found = 0
    for i, word in enumerate(vocab):
        word_present = torch.count_nonzero(pretrained_embed[word]).item()
        if word_present > 0:
            weights_matrix[i] = pretrained_embed[word]
            words_found += 1
        else:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    return torch.from_numpy(weights_matrix)

def load_ae(model, model_name = 'epoch_5_model.pth'):
    print("Loading pretrained ae...")
    model_name = 'epoch_5_model.pth'
    base_path = '/content/gdrive/MyDrive/ATGWRL/'
    saved_models_dir = os.path.join(base_path, r'saved_aes')
    model_name_path = os.path.join(saved_models_dir, model_name)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_name_path):
            model.load_state_dict(torch.load(model_name_path), strict = False)
        else:
            sys.exit("AE model path does not exist")
    else:
        sys.exit("AE path does not exist")

    return model

def create_bpe_tokenizer(tokenizer_path = "data/tokenizer-toronto.json"):
    tokenizer_path = "data/tokenizer-toronto.json"
    cwd = os.getcwd()
    tokenizer_file = os.path.join(cwd, tokenizer_path)

    if os.path.isfile(tokenizer_file):
        print("found existing tokenizer")
        tk = Tokenizer.from_file(tokenizer_file)
        print("vocab size", tk.get_vocab_size())
        all_vocab = tk.get_vocab()
        all_vocab = {k: v for k, v in sorted(all_vocab.items(), key=lambda item: item[1])}
        print("special tokens", list(all_vocab.items())[:5])
        return tk
    
    else:
        normalizer = normalizers.Sequence([NFD(), StripAccents()])
        tokenizer = Tokenizer(BPE())
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[PAD]", "EOS", "[CLS]", "[UNK]", "[SEP]", "[MASK]"])
        tokenizer.train(files=["data/bookcorpus_plain.txt"], trainer=trainer)
        tokenizer.save(tokenizer_file)
        return tokenizer
    
def tokenize_data(tokenizer, plain_path = "data/bookcorpus_plain.txt", ids_path = "data/bookcorpus_ids.txt"):

    cwd = os.getcwd()
    ids_file = os.path.join(cwd, ids_path)

    if os.path.isfile(ids_file):
        print("data already tokenized, moving on")
        return
    else:
        if not os.path.isfile(plain_path):
            print("load dataset ...")
            dset = load_dataset("bookcorpus")
    
            with open("bookcorpus_plain.txt", "w") as f:
                for element in tqdm(dset['train']):
                    f.write(element['text'] + "\n")
                f.close()
    
        f = open(plain_path, "r")
        g = open(ids_path, "w")
        cnt = 0
    
        while True:
            line = f.readline()
            line = line.rstrip()
            if not line:
                break
            cnt += 1
            encoded_line = tokenizer.encode(line)
            ids = [str(x) for x in encoded_line.ids]
            if len(ids) > MAX_TOKEN_LENGTH:
                continue
            ids_as_str = ' '.join(ids)
            g.write(ids_as_str + "\n")
    
        print("Files created. Found {} sentences", cnt)
        return

def my_plot(epochs, re_list):
    re_list = np.array(re_list)
    #encoder_loss_list = np.array(encoder_loss_list)
    #kl_div_list = np.array(kl_div_list)
    #kl_weight_list = np.array(kl_weight_list)

    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'plotted_losses')
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = 'Plotted loss after ' + str(epochs) + 'batches.png'
    final_directory = os.path.join(directory, filename)

    temp = epochs
    epochs = []
    for i in range(temp):
        epochs.append(i)

    epochs = np.array(epochs)
    
    plt.plot(epochs, re_list, label = 'reconstruction error')
    #plt.plot(epochs, encoder_loss_list, label = 'encoder loss (scaled by 5 for visual)')
    #plt.plot(epochs, kl_weight_list, label = 'kl weight')
    
    #plt.yscale('log')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Loss plotted over ' + str(temp) + ' batches')
    plt.legend()
    plt.savefig(final_directory, dpi=300)
    plt.close()

# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# from https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
def update(existingAggregate, newValues):
    if isinstance(newValues, (int, float, complex)):
        # Handle single digits.
        newValues = [newValues]
    (count, mean, M2) = existingAggregate
    count += len(newValues) 
    # newvalues - oldMean
    delta = np.subtract(newValues, [mean] * len(newValues))
    mean += np.sum(delta / count)
    # newvalues - newMeant
    delta2 = np.subtract(newValues, [mean] * len(newValues))
    M2 += np.sum(delta * delta2)

    return (count, mean, M2)

def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)

def model_usage_by_layer(time_list):
    total_time = sum(time_list)

    decoding_time = time_list[0] / total_time
    reparam_time = time_list[1] / total_time
    code_layer_time = time_list[2] / total_time
    encoder_plus_dropout_time = time_list[3] / total_time
    embedding_time = time_list[4] / total_time

    print("Model times by percentage: Embedding {} | Encoder {} | Code Layer {} | Reparam {} | Decoding {}".format(embedding_time, encoder_plus_dropout_time, code_layer_time, reparam_time, decoding_time))
                   
def sub_ner_tokens(sentence, nlp):
    # replace Person, Organisation, and location tokens
    # with placeholder tokens for easier processing
    common_nes ={"PERSON": "PERSON_token", "ORG": "ORG_token", "GPE": "GPE_token"}
    #sentence = WhitespaceTokenizer().tokenize(sentence)

    # casing required for NER
    s_upper = truecase.get_true_case(sentence)
    s_upper_tokenized = TreebankWordTokenizer().tokenize(s_upper)

    # +1 cause spacy requires joining str with spaces
    word_lens = [len(x) + 1 for x in s_upper_tokenized] 
    word_lens[-1] = word_lens[-1] - 1
    character_pos = [sum(word_lens[:i]) for i in range(len(word_lens))]

    s_upper = ' '.join(s_upper_tokenized)

    doc = nlp(s_upper)

    labels = [ent.label_ for ent in doc.ents]
    entity_text = [ent.text for ent in doc.ents]
    label_freq = []

    relevant_idx = []
    relevant_ne = []

    if labels:
        for idx, label in enumerate(labels):
            token = common_nes.get(label)
            if token:
                relevant_idx.append(idx)
                relevant_ne.append(token)

    # return unaltered sentence if no labels found
    if not relevant_ne:
        return sentence

    token_start_idx = []
    for idx, ent in enumerate(doc.ents):
        if idx in relevant_idx:
            token_start_idx.append(character_pos.index(ent.start_char))

    temp = list(range(len(relevant_ne)))
    random.shuffle(temp)

    for idx, ne, rd_idx in zip(token_start_idx, relevant_ne, temp):
        s_upper_tokenized[idx] = ne + str(rd_idx)

    sentence = ' '.join(s_upper_tokenized)
    sentence = sentence.lower()

    return sentence

def save_model(epoch, model):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_aes')
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = 'epoch_' + str(epoch) + '_model_' + model.name +  '.pth'
    final_directory = os.path.join(directory, filename)
    torch.save(model.state_dict(), final_directory)

def word_deletion(batch, prob = 0.3):
    new_batch = []
    for sentence in batch:
        new_sentence = []
        for word in sentence:
            rand_float = random.uniform(0, 1)
            if rand_float > prob:
                new_sentence.append(word)
        if not new_sentence:
            new_sentence.append(sentence[0])

        new_batch.append(new_sentence)
    
    return new_batch

def real_lengths(unpadded_list):
    sent_lens = [len(i) for i in unpadded_list]
    sent_lens = [max(min(x, MAX_SENT_LEN), 0) for x in sent_lens]
    return sent_lens

def pad_batch(batch):
    list_len = [len(i) for i in batch]
    max_len = MAX_SENT_LEN
    padded_batch = []
    for element in batch:
        element = element[:max_len]
        len_dif = max_len - len(element)
        if len_dif > 0:
            element = element + [PAD_ID] * len_dif
        padded_batch.append(element)
    return padded_batch

def pad_batch_and_add_EOS(batch):
    list_len = [len(i) for i in batch]
    max_len = MAX_SENT_LEN
    padded_batch = []
    for element in batch:
        element = element + [EOS_ID]
        if max_len - len(element) > 0:
            element.extend([PAD_ID] * (max_len - len(element)))
        element = element[:max_len]
        padded_batch.append(element)
    return padded_batch
    
def reformat_decoded_batch(decoded_batch, pad_id):
    decoded_batch = torch.transpose(decoded_batch, 1, 0)
    decoded_batch = decoded_batch.tolist()
    max_len = MAX_SENT_LEN
    reformatted_batch = []
    for element in decoded_batch:
        try:
            first_zero = element.index(pad_id)
        except:
            first_zero = len(element)

        element = element[:first_zero]
        if len(element) < max_len:
            element = element + [0] * (max_len - len(element))
        reformatted_batch.append(element)

    return reformatted_batch

def average_over_nonpadded(accumulated_loss, weights, seqlen_dim):
    total_size = torch.sum(weights, dim = seqlen_dim)
    total_size += 1e-12  # avoid division by 0 for all-0 weights.
    return accumulated_loss / total_size

def return_weights(real_lens):
    # the sentence and the first padding token
    # weighted as 1 (model rewarded for ending sentence)
    max_len = MAX_SENT_LEN

    batch_weights = []
    for length in real_lens:
        length = length + 1 # add 1 to include EOS token
        weights = [1] * length + (max_len - length) * [0]
        weights = weights[:max_len]
        batch_weights.append(weights)

    return batch_weights

def load_vocab(vocab_path, max_size = 40_000):
    count = 0
    vocab = {}
    vocab_file = open(vocab_path, 'r')
    while True:
        line = vocab_file.readline()
        line = line.rstrip()
        if not line:
            break
        vocab[line] = count
        count += 1
        if count > max_size:
            break
    revvocab = {v: k for k, v in vocab.items()}
    return vocab, revvocab

def load_names(names_path):
    name_file = open(names_path, 'r')
    name_list = []
    while True:
        line = name_file.readline()
        line = line.rstrip()
        line = line.lower()
        if not line:
            break
        name_list.append(line)
    return name_list

def load_data_from_file(data_path, max_num_of_sents = None, debug = False, skip_sents = None):
    cwd = os.getcwd()
    data_path = os.path.join(cwd, data_path)
    
    if max_num_of_sents:
        print("loading data ids... (only {} sentences)".format(max_num_of_sents))
    else:
        print("loading data ids... (est time 3 mins)")
    t1 = time.time()
    data = []
    data_file = open(data_path, 'r')
    
    counter = 0
    skip_counter = 0

    while True:
        line = data_file.readline()
        if not line:
            break
        if skip_sents:
            skip_counter += 1
            if skip_counter < skip_sents:
                continue
        line = line[2:-2]
        line = line.replace(" ", "")
        line = line.split(",")
        line = [int(x) for x in line]
        data.append(line)
        if max_num_of_sents:
            counter += 1
            if counter >= max_num_of_sents:
                break
        
    data_file.close()
    t2 = time.time()
    print("loading data ids took {:.2f} seconds".format(t2-t1))
    return data
        
def load_data_and_create_vocab(dataset = "bookcorpus", word_freq_cutoff = 5, vocab_path = "vocab.txt", names_path = "names.txt"):

    if os.path.isfile(vocab_path) and sum(1 for line in open(vocab_path)) > 10_000:
        x = None
        print("Found existing vocab file, loading now...")
        vocab, revvocab = load_vocab(vocab_path)
        print("Vocab size: ", len(vocab))
        return vocab, revvocab, x
    else:
        import enchant
        american_words = enchant.Dict("en_US")
        dset = load_dataset(dataset)
        human_names = load_names(names_path)
        human_names = set(human_names)
        nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

        special_tokens = []

        for key in common_nes.keys():
            for i in range(10):
                special_tokens.append(key + str(i))

        word_freqs = {}
        print("creating vocabulary...")
        for element in tqdm(dset['train']):
            sentence = sub_ner_tokens(element['text'], nlp)
            for word in sentence:
                if word:
                    word = re.sub("^[0-9]+$", "num000", word)
                    if word not in word_freqs:
                        word_freqs[word] = 1
                    else:
                        word_freqs[word] += 1

        word_freqs = {k: v for k, v in word_freqs.items()}
        unfiltered_len = len(word_freqs)
        print("number of words (without filtering)", unfiltered_len)

        word_freqs = {k: v for k, v in word_freqs.items() if v >= word_freq_cutoff}

        print("number of words after deleting words less frequent words", len(word_freqs))

        sorted_word_freqs = {k: v for k, v in sorted(word_freqs.items(), key=lambda item: item[1], reverse=True)}

        with open("vocab.txt", "w") as f:
            f.write("PAD\n")
            f.write("EOS\n")
            f.write("BOS\n")
            f.write("UNK\n")
            for key in sorted_word_freqs:
                f.write(key+ "\n")
        f.close()

        vocab, revvocab = load_vocab(vocab_path)
        return vocab, revvocab, dset

def prepare_data(dset, vocab, vocab_path = "vocab.txt", data_ids = "bookcorpus_ids.txt", data_plain = "bookcorpus_plain.txt", names_path = "names.txt", max_sent_len = 20):

    if os.path.isfile(data_ids) and os.path.getsize(data_ids) > 0:
        print("Found existing data file. Size = ", sum(1 for line in open(data_ids)))
        return sum(1 for line in open(data_ids))
    else:
        print("Creating data files with name {} and {}".format(data_ids, data_plain))
        data = []
        vocab_rejection = 0
        sent_len_rejection = 0
        cnt = 0
        punctuation = ["!", "?", ".", ",", ";", ":", "'", "-"]
        human_names = load_names(names_path)
        human_names = set(human_names)
        dset = load_dataset("bookcorpus")
        nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

        with open(data_ids, "w") as f, open(data_plain, "w") as g:
            for element in tqdm(dset['train']):
                sentence = sub_ner_tokens(element['text'], nlp)
                line_ids = []
                line_words = []
                sentence_length = 0
                add_sentence = True
                for word in sentence:
                    if not word:
                        continue
                    word = re.sub("^[0-9]+$", "num000", word)
                    if word not in vocab:
                        add_sentence = False
                        vocab_rejection +=1
                        break
                    if word not in punctuation:
                        sentence_length += 1
                    if sentence_length > max_sent_len:
                        add_sentence = False
                        sent_len_rejection += 1
                        break
                    line_ids.append(vocab[word])
                    line_words.append(word)
                if add_sentence and len(line_ids) > 1:
                    f.write(" ".join(str(line_ids)))
                    f.write("\n")
                    g.write(" ".join(line_words))
                    g.write("\n")
                    data.append(line_ids)
                cnt += 1
        f.close()
        g.close()
        print("vocabrejection {}, sentlenrejection {}".format(vocab_rejection, sent_len_rejection))
        print("Data file created. Size = ", sum(1 for line in open(data_ids)))
        return sum(1 for line in open(data_ids))

def yieldBatch(batch_size, data):
    # lazy iterator for batches
    # PARAMS: batch_size, data
    random.shuffle(data)
    sindex=0
    eindex=batch_size
    while eindex < len(data):
        batch = data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        
        yield batch

def sample_batch(batch_size, data):
    # sampling batch from data with replacement
    # PARAMS: batch_size, data
    sample_num = random.randint(0, len(data) - batch_size - 1)
    data_batch = data[sample_num:sample_num+batch_size]
    return data_batch