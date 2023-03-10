from datasets import load_dataset
from nltk.tokenize import WhitespaceTokenizer, TreebankWordTokenizer
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
    print("voab size:", config.vocab_size)
    print("batch_size:", config.ae_batch_size)
    print("latent dimension:", config.latent_dim)
    print("layer normalisation:", config.layer_norm)
    print("word embedding size:", config.word_embedding)
    print("dropout probability:", config.dropout_prob)
    print("learning rate:", config.ae_learning_rate)
    print("ae betas:", *config.ae_betas)
    if model.name == "default_autoencoder":
        print("using pretrained embedding:", config.pretrained_embedding)
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

    filename = 'epoch_' + str(epoch) + '_model.pth'
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

def load_vocab(vocab_path, max_size = 20_000):
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

def load_data_from_file(data_path, max_num_of_sents = None, debug = False, retain_validation = True):
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

    while True:
        line = data_file.readline()
        if not line:
            break
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