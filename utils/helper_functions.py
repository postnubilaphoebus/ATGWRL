from datasets import load_dataset
from nltk.tokenize import WhitespaceTokenizer
import os.path
from tqdm import tqdm
import regex as re
import numpy as np
import time
import torch

CONTRACTION_MAP = {"ain't": "is not", "aren't": "are not", "can't": "cannot",
                   "can't've": "cannot have", "'cause": "because", "could've": "could have",
                   "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
                   "he'll've": "he he will have", "he's": "he is", "how'd": "how did",
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                   "I'll've": "I will have", "I'm": "I am", "I've": "I have",
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                   "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                   "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                   "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                   "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                   "she's": "she is", "should've": "should have", "shouldn't": "should not",
                   "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                   "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                   "they'll've": "they will have", "they're": "they are", "they've": "they have",
                   "to've": "to have", "wasn't": "was not", "we'd": "we would",
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                   "we're": "we are", "we've": "we have", "weren't": "were not",
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                   "what's": "what is", "what've": "what have", "when's": "when is",
                   "when've": "when have", "where'd": "where did", "where's": "where is",
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                   "who's": "who is", "who've": "who have", "why's": "why is",
                   "why've": "why have", "will've": "will have", "won't": "will not",
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                   "you'll've": "you will have", "you're": "you are", "you've": "you have", "here's": "here is",
                   "here're": "here are", "'d": "had", "'s": "is", "n't": "not"}

def save_model(epoch, model):
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'saved_vaes')
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = 'epoch_' + str(epoch) + '_model.pth'
    final_directory = os.path.join(directory, filename)
    torch.save(model.state_dict(), final_directory)

def real_lengths(unpadded_list):
    sent_lens = [len(i) for i in unpadded_list]
    return sent_lens

def pad_batch(batch):
    list_len = [len(i) for i in batch]
    max_len = max(list_len)
    padded_batch = []
    for element in batch:
        element.extend([0] * (max_len - len(element)))
        padded_batch.append(element)

    return padded_batch
    
def reformat_decoded_batch(decoded_batch, pad_id):
    decoded_batch = torch.transpose(decoded_batch, 1, 0)
    decoded_batch = decoded_batch.tolist()
    max_len = len(decoded_batch[0])
    reformatted_batch = []
    for element in decoded_batch:
        # we include the first PAD as a form of EOS_id
        first_zero = element.index(pad_id) + 1
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
    max_len = max(real_lens)

    batch_weights = []
    for length in real_lens:
        length = length + 1 # add 1 to include pad token as EOS
        weights = [1] * length + (max_len - length) * [0]
        weights = weights[:max_len]
        batch_weights.append(weights)

    return batch_weights

def load_vocab(vocab_path):
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

def load_data_from_file(data_path):
    print("loading data ids... (est time 3 mins)")
    t1 = time.time()
    data = []
    data_file = open(data_path, 'r')

    debug = True

    if debug:
        debugging_idx = 0
        print("debugging load (only 100k)")

    while True:
        line = data_file.readline()
        line = line[1:-2]
        if not line:
            break
        line = line.replace(" ", "")
        line = line.split(",")
        line = [int(x) for x in line]
        data.append(line)
        if debug:
            debugging_idx += 1
            if debugging_idx > 100_000:
                break
    data_file.close()
    t2 = time.time()
    print("loading data ids took {:.2f} seconds".format(t2-t1))
    return data

def pychant_lookup(word, human_names, american_words):
    if word == "num000" or word in human_names or american_words.check(word) or american_words.check(word.upper()):
        return True
    if CONTRACTION_MAP.get(word):
        return True
    return False
        
def load_data_and_create_vocab(dataset = "bookcorpus", word_freq_cutoff = 5, vocab_path = "vocab.txt", names_path = "names.txt"):

    if os.path.isfile(vocab_path) and sum(1 for line in open(vocab_path)) > 20_000:
        x = None
        print("Found existing vocab file. Size = ", sum(1 for line in open(vocab_path)))
        vocab, revvocab = load_vocab(vocab_path)
        return vocab, revvocab, x
    else:
        import enchant
        american_words = enchant.Dict("en_US")
        dset = load_dataset(dataset)
        human_names = load_names(names_path)
        human_names = set(human_names)
        
        word_freqs = {}
        print("creating vocabulary...")
        for element in tqdm(dset['train']):
            tk = WhitespaceTokenizer().tokenize(element['text'])
            for word in tk:
                if word:
                    word = re.sub("^[0-9]+$", "num000", word)
                    if word not in word_freqs:
                        word_freqs[word] = 1
                    else:
                        word_freqs[word] += 1

        word_freqs = {k: v for k, v in word_freqs.items() if pychant_lookup(k, human_names, american_words)}
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

        with open(data_ids, "w") as f, open(data_plain, "w") as g:
            for element in tqdm(dset['train']):
                tk = WhitespaceTokenizer().tokenize(element['text'])
                line_ids = []
                line_words = []
                sentence_length = 0
                add_sentence = True
                for word in tk:
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
    sindex=0
    eindex=batch_size
    while eindex < len(data):
        batch = data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        
        yield batch
    
