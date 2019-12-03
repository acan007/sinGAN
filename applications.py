import sys
import os
import click
import torch
import matplotlib.pyplot as plt

from chanye._settings import set_numpy_precision, seed_random
from utils import get_config
from model import SinGAN
from paint_to_image import Paint2Image
from harmonization import Harmonization
from editing import Editing
from super_resolution import SuperResolution

"""
1. super_resolution 에서 train()하게
2. 나머지는 model 에서 train() 하고 인퍼런스만 딴데서
3. 
"""


@click.command()
@click.option('--config', type=str, default='./config/random_sample.yaml', help='Path to the config file.')
@click.option('--dataset_path', type=str, default='./assets/Input', help='Path to root dataset path')
@click.option('--mode', type=str, required=True,
              help='which application [paint2image | editing | harmonization | random_sample | SR]')
def main(config, dataset_path, mode):
    seed_random()
    set_numpy_precision()

    # model
    if mode == 'random_sample':
        model_type = SinGAN
    elif mode == 'paint2image':
        model_type = Paint2Image
    elif mode == 'editing':
        model_type = Editing
    elif mode == 'harmonization':
        model_type = Harmonization
    elif mode == 'SR':
        model_type = SuperResolution
    else:
        print("Invalid Parameter")
        raise ValueError

    model = model_type(get_config(config), dataset_path)
    model.test_samples(save=True)


if __name__ == '__main__':
    main()
