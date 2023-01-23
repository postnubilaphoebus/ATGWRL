from vae import VariationalAutoEncoder
from nltk.translate.bleu_score import sentence_bleu
import torch
import os
import sys
import random
import time
from utils.helper_functions import load_data_and_create_vocab, \
                                   prepare_data, \
                                   yieldBatch, \
                                   load_data_from_file, \
                                   real_lengths, \
                                   pad_batch, \
                                   return_weights, \
                                   save_model, \
                                   average_over_nonpadded, \
                                   reformat_decoded_batch

def load_vae(config):
    print("Loading pretrained vae...")
    model_5 = 'epoch_50_model.pth'
    base_path = '/content/gdrive/MyDrive/ATGWRL/'
    saved_models_dir = os.path.join(base_path, r'saved_vaes')
    model_10_path = os.path.join(saved_models_dir, model_5)
    model = VariationalAutoEncoder(config)
    model.to(config.device)

    if os.path.exists(saved_models_dir):
        if os.path.isfile(model_10_path):
            #print("Loading pretrained gen...")
            model.load_state_dict(torch.load(model_10_path), strict = False)
        else:
            sys.exit("VAE model path does not exist")
    else:
        sys.exit("VAE path does not exist")

    return model

def load_100k(data_path):
    t1 = time.time()
    data = []
    data_file = open(data_path, 'r')

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
        debugging_idx += 1
        if debugging_idx > 100_000:
            break
    data_file.close()
    t2 = time.time()
    print("loading data ids took {:.2f} seconds".format(t2-t1))
    return data


def test(config):
    vocab, revvocab, dset = load_data_and_create_vocab()
    config.vocab_size = len(revvocab)
    validation_data = load_100k("bookcorpus_ids.txt")
    random.seed(10)
    model = load_vae(config)
    model.eval()

    one_gram_bleu = 0
    two_gram_bleu = 0
    three_gram_bleu = 0
    four_gram_bleu = 0

    number_of_sents = 0
    print("Testing model on data it has seen...")

    for batch_idx, batch in enumerate(yieldBatch(config.batch_size, validation_data)):
        original_lens_batch = real_lengths(batch)
        padded_batch = pad_batch(batch)
        padded_batch = torch.LongTensor(padded_batch).to(model.device)

        encoded, z_mean, z_log_var, decoded_tokens, decoded_logits, time_list = model(padded_batch, original_lens_batch, False)
        
        decoded_tokens = reformat_decoded_batch(decoded_tokens, 0)
        padded_batch = padded_batch.tolist()

        for decoded, target in zip(decoded_tokens, padded_batch):
            try:
                first_zero = target.index(0) + 1
                decoded = decoded[:first_zero]
                target = target[:first_zero]
            except:
                pass
            dec_sent = [revvocab[x] for x in decoded]
            target_sent = [revvocab[x] for x in target]
            
            one_gram_bleu += sentence_bleu(target_sent, dec_sent,   weights=(1, 0, 0, 0))
            two_gram_bleu += sentence_bleu(target_sent, dec_sent,   weights=(0, 1, 0, 0))
            three_gram_bleu += sentence_bleu(target_sent, dec_sent, weights=(0, 0, 1, 0))
            four_gram_bleu += sentence_bleu(target_sent, dec_sent,  weights=(0, 0, 0, 1))
            number_of_sents += 1
            
            if batch_idx % 1000 == 0:
                print("dec_sent", dec_sent)
                print("target_sent", target_sent)
                one_gram_bleu /= number_of_sents
                two_gram_bleu /= number_of_sents
                three_gram_bleu /= number_of_sents
                four_gram_bleu /= number_of_sents
                print("1-gram {}, 2-gram {}, 3-gram {}, 4-gram {}".format(one_gram_bleu, two_gram_bleu, three_gram_bleu, four_gram_bleu))
                
    print("final bleu scores")
    one_gram_bleu /= number_of_sents
    two_gram_bleu /= number_of_sents
    three_gram_bleu /= number_of_sents
    four_gram_bleu /= number_of_sents

    print("1-gram {}, 2-gram {}, 3-gram {}, 4-gram {}".format(one_gram_bleu, two_gram_bleu, three_gram_bleu, four_gram_bleu))




