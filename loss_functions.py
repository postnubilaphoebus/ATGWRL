import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from utils.helper_functions import average_over_nonpadded, \
                                   yieldBatch, \
                                   pad_batch, \
                                   real_lengths, \
                                   pad_batch_and_add_EOS, \
                                   return_weights, \
                                   reformat_decoded_batch, \
                                   rouge_and_bleu

def encoder_loss(model, encoded, re_embedded, x_lens, loss_fn = None):
    # as given by Oshri and Khandwala in: 
    # There and Back Again: Autoencoders for Textual Reconstruction:

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        c_prime, _ = model.encoder(re_embedded, x_lens, skip_embed = True)
        
    c_base = encoded
                
    encoder_loss = []
                
    for target, inp in zip(c_base, c_prime):
        encoder_loss.append(loss_fn(inp, target))
                    
    encoder_loss = torch.stack((encoder_loss))
    encoder_loss = torch.mean(encoder_loss)
    
    return encoder_loss

def reconstruction_loss(weights, targets, decoded_logits, loss_fn = None, label_smoothing = None):

    if loss_fn is None:
        if label_smoothing:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing = label_smoothing)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    weights = torch.transpose(weights, 1, 0)
    targets = torch.transpose(targets, 1, 0)
    reconstruction_error = []

    for weight, target, logit in zip(weights, targets, decoded_logits):
        ce_loss = loss_fn(logit, target)
        ce_loss = torch.mean(ce_loss, dim = -1)
        reconstruction_error.append(ce_loss * weight)

    reconstruction_error = torch.stack((reconstruction_error))
    reconstruction_error = torch.sum(reconstruction_error, dim = 0) # sum over seqlen
    reconstruction_error = average_over_nonpadded(reconstruction_error, weights, 0) # av over seqlen
    reconstruction_error = torch.mean(reconstruction_error) # mean over batch

    return reconstruction_error

def autoencoder_bleu(decoded_logits, padded_batch, revvocab):
    smoothie = SmoothingFunction().method1
    m = torch.nn.Softmax(dim = -1)
    decoded_tokens = torch.argmax(m(decoded_logits), dim = -1)
    decoded_tokens = reformat_decoded_batch(decoded_tokens, 0)
    padded_batch = padded_batch.tolist()
    number_of_sents = 0
    bleu4 = 0
    for decoded, target in zip(decoded_tokens, padded_batch):
        try:
            first_zero = target.index(0)
            decoded = decoded[:first_zero+1]
            target = target[:first_zero]
            target = target + [1] # + EOS_ID
        except:
            pass
        dec_sent = [revvocab[x] for x in decoded]
        target_sent = [revvocab[x] for x in target]

        bleu4 += sentence_bleu(target_sent, dec_sent, smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
        number_of_sents += 1
    bleu4 /= number_of_sents
    return round(bleu4, 4)
        

def validation_set_acc(config, model, val_set, revvocab):
    model.eval()
    re_list = []
    bleu_list = []
    for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, val_set)):
        original_lens_batch = real_lengths(batch)
        padded_batch = pad_batch(batch)
        targets = pad_batch_and_add_EOS(batch)
        weights = return_weights(original_lens_batch)

        weights = torch.FloatTensor(weights).to(model.device)
        padded_batch = torch.LongTensor(padded_batch).to(model.device)
        targets = torch.LongTensor(targets).to(model.device)
        
        with torch.no_grad():
            if model.name == "variational_autoencoder":
                _, _, decoded_logits = model(padded_batch, original_lens_batch)
            else:
                decoded_logits = model(padded_batch, original_lens_batch)

            reconstruction_error = reconstruction_loss(weights, targets, decoded_logits)
            bleu4 = autoencoder_bleu(decoded_logits, padded_batch, revvocab)
            
        re_list.append(reconstruction_error.item())
        bleu_list.append(bleu4)
        
    val_error = sum(re_list) / len(re_list)
    bleu_score = sum(bleu_list) / len(bleu_list)
    print("val_error", val_error)
    print("bleu 4 score", bleu_score)
    model.train()
    
    return val_error, bleu_score