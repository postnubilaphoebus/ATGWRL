import torch
import utils.config as config
from train_ae import train
from test_ae import test
from train_gan import train_gan

def main(*args, **kwargs):

    log_dict = train(config)
    train_gan(config)

if __name__ == "__main__":
    main(config)
