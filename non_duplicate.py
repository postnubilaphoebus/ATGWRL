from datasets import load_dataset
from nltk.tokenize import WhitespaceTokenizer
import os.path
import torchtext
from tqdm import tqdm
import regex as re
import torch

def remove_special_chars(text):
    cleaned_text = re.sub(r"[^\w\s!?.;,:'\-]", "", text)
    cleaned_text = re.sub(r'(?<=\w)\.(?=\w)', '', cleaned_text)
    cleaned_text = cleaned_text.replace("...", ".")
    cleaned_text = cleaned_text.replace("''", "")
    cleaned_text = cleaned_text.replace('""','')
    cleaned_text = cleaned_text.rstrip("-")
    return cleaned_text

def load_vocab(vocab_path, max_size = None):
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
        if max_size:
            if count > max_size:
                break
    revvocab = {v: k for k, v in vocab.items()}
    return vocab, revvocab

def read_file(file_path):
    data_file = open(file_path, 'r')
    data = []
    while True:
        line = data_file.readline()
        if not line:
            break
        data.append(line)
    return data

def load_and_filter_data(dataset = "bookcorpus", word_freq_cutoff = 5, vocab_path = "vocab.txt", filter_duplicates = True, cur_dir = None):

    if not cur_dir:
        cur_dir = os.getcwd()
    dset = load_dataset(dataset)
    print("preprocessing data...")
    data = []
    for element in tqdm(dset['train']):
        sentence = remove_special_chars(element['text'])
        data.append(sentence)

    if filter_duplicates:
        print("Data length before removing duplicates", len(data))
        data = list(set(data))
        print("Data length after removing duplicates", len(data))

    with open(os.path.join(cur_dir, "filtered_data.txt"), "w") as f:
        for sent in data:
            f.write(sent + "\n")
    f.close()

def create_vocab(src_path, vocab_path, word_freq_cutoff = 5, cur_dir = None):
    if not cur_dir:
        cur_dir = os.getcwd()
    print("reading in file...")
    data = read_file(src_path)
    print("read file, producing vocab...")
    word_freqs = {}
    for sentence in tqdm(data):
        sentence = WhitespaceTokenizer().tokenize(sentence)
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

    with open(vocab_path, "w") as f:
        f.write("PAD\n")
        f.write("EOS\n")
        f.write("BOS\n")
        f.write("UNK\n")
        for key in sorted_word_freqs:
            f.write(key+ "\n")
    f.close()

def save_data_with_vocab(vocab_path, src_path, trgt_path, glove, vocab_size):
    vocab, revvocab = load_vocab(vocab_path, vocab_size)
    data = read_file(src_path)
    max_sent_len = 28
    cnt = 0
    vocab_rejection = 0
    sent_len_rejection = 0
    with open(trgt_path, "w") as f:
        for sentence in tqdm(data):
            line_ids = []
            sentence_length = 0
            add_sentence = True
            sentence = WhitespaceTokenizer().tokenize(sentence)
            for word in sentence:
                if word.isspace() == True or not word:
                    continue
                word = re.sub("^[0-9]+$", "num000", word)
                sentence_length+=1
                if sentence_length > max_sent_len:
                    add_sentence = False
                    sent_len_rejection += 1
                    break
                if word not in vocab:
                    add_sentence = False
                    vocab_rejection += 1
                    break
                if word != "num000":
                    word_present = torch.count_nonzero(glove[word]).item()
                    if word_present == 0:
                        add_sentence = False
                        vocab_rejection += 1
                        break         
                line_ids.append(vocab[word])
            if add_sentence and len(line_ids) > 1:
                cnt+=1
                f.write(" ".join(str(line_ids)))
                f.write("\n")
    print("Wrote {} sentences to file", cnt)
    print("Vocab rejection {}, sentlen rejection {}".format(vocab_rejection, sent_len_rejection))
    f.close()
    
if __name__ == "__main__":
    #load_and_filter_data()
    src_path = "/Users/lauridsstockert/Desktop/test_new_models/filtered_data.txt"
    vocab_path = "/Users/lauridsstockert/Desktop/test_new_models/vocab.txt"
    trgt_path = "/Users/lauridsstockert/Desktop/test_new_models/corpus_v40k_ids.txt"
    glove = torchtext.vocab.GloVe(name='twitter.27B', dim=100) # 27B is uncased
    #create_vocab(src_path, vocab_path)
    save_data_with_vocab(vocab_path, src_path, trgt_path, glove, 40_000)
    
