import torch
import utils.config as config
from train_vae import train

def main(*args, **kwargs):

    log_dict = train(config)

if __name__ == "__main__":
    main(config)