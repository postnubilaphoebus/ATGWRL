import torch
import utils.config as config
from train_ae import train as train_ae
from train_gan import train_gan
from test_ae import test
from train_vae import train as train_vae
from test_gan import test as test_gan

def main(*args, **kwargs):

    log_dict = train_ae(config)
    #train_gan(config)
    #test(config)
    #test_gan(config)

if __name__ == "__main__":
    main(config)