import sys
import os
import click
import torch
import matplotlib.pyplot as plt

from utils import get_config, set_numpy_precision, seed_random
from sinGAN import SinGAN
from paint_to_image import Paint2Image
from harmonization import Harmonization
from editing import Editing
from super_resolution import SuperResolution


@click.command()
@click.option('--config', type=str, default='./config/random_sample.yaml', help='Path to the config file.')
@click.option('--mode', type=str, default='random_sample',
              help='which application [paint2image | editing | harmonization | random_sample | SR]')
@click.option('--resume', default=False, help='whether resume and train')
def main(config, mode, resume):
    seed_random()
    set_numpy_precision()

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

    model = model_type(get_config(config))
    if resume:
        model.resume_train()

    model.train()


if __name__ == '__main__':
    main()
