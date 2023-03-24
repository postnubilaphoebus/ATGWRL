from models import AutoEncoder, CNNAutoEncoder, VariationalAutoEncoder
import os
import torch
import sys
import utils.config as config
from utils.helper_functions import load_data_from_file, \
                                   yieldBatch, \
                                   real_lengths, \
                                   pad_batch, \
                                   update, \
                                   finalize


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
        elif model_name == "variational_autoencoder":
            model = VariationalAutoEncoder(config, weights_matrix)
            model = model.apply(VariationalAutoEncoder.init_weights)
            model.to(model.device)
        else:
            sys.exit("no valid model name provided")
        model.load_state_dict(torch.load(model_path, map_location = model.device), strict = False)
    else:
        sys.exit("ae model path does not exist")
    return model

def distribution_constraint(fitted_distribution, mini_batch, scaling_factor = 1.0):
    # [H, B] 
    batch_mean = torch.mean(mini_batch, dim = 1)
    batch_sigma = torch.std(mini_batch, dim = 1)
    batch_stats = torch.stack((batch_mean, batch_sigma))
    
    cnt = 0
    constraint_sum = 0
    for real_stats, gen_stats in zip(fitted_distribution, batch_stats):
        mu_diff = torch.abs(gen_stats[0] - real_stats[0])
        sigma_diff = torch.abs(gen_stats[1] - real_stats[1])
        constraint_sum += (mu_diff + sigma_diff)
        cnt += 1
        
    constraint_loss = constraint_sum / cnt
    constraint_loss /= mini_batch.size(1)
    return scaling_factor * constraint_loss

def distribution_fitting(config, 
                         model_name = "cnn_autoencoder", 
                         model_path = "/Users/lauridsstockert/Desktop/test_new_models/saved_aes/epoch_5_model_cnn_autoencoder.pth", 
                         data_path = "corpus_v20k_ids.txt",
                         vocab_path = "vocab_20k.txt",
                         validation_size = 10_000):
    config.vocab_size = 20_000
    config.encoder_dim = 600
    config.word_embedding = 300
    model = load_model(config, model_name, model_path)
    model.eval()

    print("loading data: {} and vocab: {}".format(data_path, vocab_path)) 
    data = load_data_from_file(data_path, 20_000)
    val, all_data = data[:validation_size], data[validation_size:]
    data_len = len(all_data)
    print("Loaded {} sentences".format(data_len))

    config.ae_batch_size = 1000
    initial_aggregate = (0, 0, 0) # mean, variance, samplevariance
    result_batch = [initial_aggregate] * config.latent_dim

    for batch_idx, batch in enumerate(yieldBatch(config.ae_batch_size, all_data)):
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
            # [B, H] -> [H, B]
            z = torch.transpose(z, 1, 0)
            z = z.cpu().detach().numpy()
            for idx, hidden in enumerate(z):
                result_batch[idx] = update(result_batch[idx], hidden)

    # finalising mu and sigma
    for idx, elem in enumerate(result_batch):
        result_batch[idx] = finalize(elem)

    
    distribution_mean = [x[0] for x in result_batch]
    distribution_variance = [x[1] for x in result_batch]
    distribution_mean = torch.FloatTensor(distribution_mean)
    distribution_variance = torch.FloatTensor(distribution_variance)
    fitted_distribution = torch.stack((distribution_mean, distribution_variance)).to(config.device)
    z = torch.FloatTensor(z).to(config.device)
    return fitted_distribution, z

if __name__ == "__main__":
    fitted_distribution, z = distribution_fitting(config)

    loss = distribution_constraint(fitted_distribution, z)
    print("loss", loss)
    


