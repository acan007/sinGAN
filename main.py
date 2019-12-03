import sys

import numpy as np
import os
import click
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from chanye._settings import set_dev_location, get_dataset_path, set_numpy_precision, seed_random

from utils import get_config
from model import SinGAN


@click.command()
@click.option('--config', type=str, default='./config/random_sample.yaml', help='Path to the config file.')
@click.option('--location', type=str, required=True, help='dev env [macbook | server | home]')
@click.option('--resume', default=False, help='whether resume and train')
def main(config, location, resume):
    seed_random()
    set_numpy_precision()
    set_dev_location(location)

    config = get_config(config)

    try:
        path_img_load = os.path.join(get_dataset_path(), config['path_data'])
        input = plt.imread(path_img_load)[:, :, :3]
    except FileNotFoundError:
        print("ERROR!! You should put input sample according to `get_dataset_path()`")
        sys.exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = SinGAN(input, config, device)
    # TODO
    # if load_path:
    #     model.load_models(load_path, resume)

    # train
    model.train()


if __name__ == '__main__':
    main()
