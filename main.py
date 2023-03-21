import torch
import utils.config as config
from train_ae import train
from train_gan import train_gan
from test_ae import test
#from train_vae import train

def main(*args, **kwargs):

    #log_dict = train(config)
    train_gan(config)
    #test(config)

if __name__ == "__main__":
    main(config)