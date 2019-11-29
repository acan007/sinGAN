import numpy as np
import os
import click
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from _settings import set_dev_location, get_dataset_path, set_numpy_precision
import _telegramer as telegramer

from utils import get_config, adjust_scale_factor_by_image
from model import SinGAN


@click.command()
@click.option('--config', type=str, default='./config/random_sample.yaml', help='Path to the config file.')
@click.option('--location', type=str, default='macbook', help='dev env [macbook | server | home]')
@click.option('--resume', default=False, help='whether resume and train')
def main(config, location, resume):
    set_dev_location(location)
    set_numpy_precision()

    config = get_config(config)

    try:
        path_img_load = os.path.join(get_dataset_path(), config['path_data'])
        input = plt.imread(path_img_load)[:, :, :3]
    except Exception as e:
        print(e)
        print("You should put input image according to `get_dataset_path()`")
    config = adjust_scale_factor_by_image(input, config)

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SinGAN(input, config, device)
    # TODO
    # if load_path:
    #     model.load_models(load_path, resume)

    # train
    model.train()

    telegramer.send_text('MUNIT LEARNING FINISHED')


if __name__ == '__main__':
    main()
