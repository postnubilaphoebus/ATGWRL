from models import AutoEncoder, CNNAutoEncoder, ExperimentalAutoencoder, VariationalAutoEncoder
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_scoring
from rouge_score import rouge_scorer
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
                                   reformat_decoded_batch, \
                                   rouge_and_bleu, \
                                   load_vocab, \
                                   return_bert_score

def load_model(config, model_name, model_path, weights_matrix = None):
    if os.path.isfile(model_path):
        if model_name == "default_autoencoder":
            model = AutoEncoder(config, weights_matrix)
            model = model.apply(AutoEncoder.init_weights)
            model.to(model.device)
        elif model_name == "cnn_autoencoder":
            model = CNNAutoEncoder(config, weights_matrix)
            model = model.apply(CNNAutoEncoder.init_weights)
            model.to(model.device)
        elif model_name == "strong_autoencoder":
            model = ExperimentalAutoencoder(config, weights_matrix)
            model = model.apply(ExperimentalAutoencoder.init_weights)
            model.to(model.device)
        elif model_name == "variational_autoencoder":
            model = VariationalAutoEncoder(config)
            model = model.apply(VariationalAutoEncoder.init_weights)
            model.to(model.device)
        else:
            sys.exit("no valid model name provided")
        model.load_state_dict(torch.load(model_path, map_location = model.device), strict = False)
    else:
        sys.exit("ae model path does not exist")
    return model

def test(config):
    location = "/data/s4184416/peregrine/saved_aes/epoch_4_model_cnn_autoencoder.pth"
    #model_path = "/Users/lauridsstockert/Desktop/test_new_models/saved_aes/epoch_4_model_cnn_autoencoder.pth"
    model_path = location
    model_name = "cnn_autoencoder"
    config.word_embedding = 100
    config.encoder_dim = 100
    #model_name = "variational_autoencoder"
    #model_path =  location
    #model_path = "/Users/lauridsstockert/Desktop/test_new_models/saved_aes/epoch_5_model_variational_autoencoder.pth"
    model = load_model(config, model_name, model_path)
    model.eval()
    loaded_sents = 10_000
    data = load_data_from_file("corpus_v40k_ids.txt", max_num_of_sents = loaded_sents)
    vocab, revvocab = load_vocab("vocab_40k.txt", 40_000)
    config.vocab_size = len(revvocab)

    scorer = rouge_scorer.RougeScorer(['rouge1', "rouge2", "rouge3", 'rouge4'], use_stemmer=True)
    score_names = ["rouge1", "rouge2", "rouge3", "rouge4", "bleu1", "bleu2", "bleu3", "bleu4", "bert score"]

    original_lens_batch = real_lengths(data)
    padded_batch = pad_batch(data)
    padded_batch = torch.LongTensor(padded_batch).to(model.device)
    step_size = 1000
    decoded_list = []
    for i in range(0, 10_000, step_size):
        decoded_logits = model(padded_batch[i:i+1000], original_lens_batch[i:i+1000])
        decoded_tokens = torch.argmax(decoded_logits, dim = -1)
        decoded_tokens = reformat_decoded_batch(decoded_tokens, 0)
        decoded_list.extend(decoded_tokens)
    padded_batch = padded_batch.tolist()

    scores = [0] * 9

    decoded_sents = []
    target_sents = []

    for decoded, target in zip(decoded_list, padded_batch):
        try:
            first_zero = target.index(0)
            decoded = decoded[:first_zero]
            target = target[:first_zero]
            target = target
        except:
            pass
        dec_sent = [revvocab[x] for x in decoded]
        target_sent = [revvocab[x] for x in target]
        dec_sent = " ".join(dec_sent)
        target_sent = " ".join(target_sent)
        decoded_sents.append(dec_sent)
        target_sents.append(target_sent)
        interim = rouge_and_bleu(dec_sent, target_sent, scorer)
        for i in range(len(interim)):
            scores[i] += interim[i]

    batched_score = 0
    cnt = 0
    for i in range(0, 10_000, step_size):
        dec = decoded_sents[i:i+1000]
        tar = target_sents[i:i+1000]
        bs = return_bert_score(dec, tar, device=config.device, batch_size=step_size)
        bs = round(bs, 4)
        batched_score += bs
        cnt +=1
    scores = [x / loaded_sents for x in scores]
    scores[-1] = batched_score / cnt
    for score, name in zip(scores, score_names):
        print("{}: {}".format(name, score))