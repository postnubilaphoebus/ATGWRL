import torch
import utils.config as config
from train_ae import train as train_ae
from train_gan import train_gan
#from test_ae import test
from train_vae import train as train_vae
from test_gan import test as test_gan

# https://pyro.ai/examples/svi_part_i.html
# for paper

def main(*args, **kwargs):

    # default_autoencoder
    # cnn_autoencoder
    # variational_autoencoder

    #log_dict = train_vae(config)
    #train_gan(config, model_name = "default_autoencoder")
    #test(config)
    test_gan(config)

if __name__ == "__main__":
    main(config)