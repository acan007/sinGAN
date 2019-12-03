import sys
import os
import click
import torch
import matplotlib.pyplot as plt

from utils import get_config, set_numpy_precision, seed_random
from model import SinGAN

# TODO
# 1. SR 에서 학습할때 z_rec 넣는 방법

"""
1. super_resolution 에서 train()하게
2. 나머지는 model 에서 train() 하고 인퍼런스만 딴데서
3. 
"""

@click.command()
@click.option('--config', type=str, default='./config/random_sample.yaml', help='Path to the config file.')
@click.option('--dataset_path', type=str, default='./assets/Input', help='Path to root dataset path')
@click.option('--resume', default=False, help='whether resume and train')
def main(config, dataset_path, resume):
    seed_random()
    set_numpy_precision()

    # model
    model = SinGAN(get_config(config), dataset_path)
    # TODO
    # if resume:
    #     model.load_models(load_path, resume)

    model.train()


if __name__ == '__main__':
    main()
