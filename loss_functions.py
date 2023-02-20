import torch
from utils.helper_functions import average_over_nonpadded

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

def reconstruction_loss(weights, targets, decoded_logits, loss_fn = None):

    if loss_fn is None:
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