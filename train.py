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


# TODO
# 1. SR 에서 학습할때 z_rec 넣는 방법

@click.command()
@click.option('--config', type=str, default='./config/random_sample.yaml', help='Path to the config file.')
@click.option('--mode', type=str, default='random_sample',
              help='which application [paint2image | editing | harmonization | random_sample | SR]')
@click.option('--dataset_path', type=str, default='./assets/Input', help='Path to root dataset path')
@click.option('--resume', default=False, help='whether resume and train')
def main(config, mode, dataset_path, resume):
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

    # model
    model = model_type(get_config(config), dataset_path)
    # TODO
    # if resume:
    #     model.load_models(load_path, resume)

    model.train()


if __name__ == '__main__':
    main()
