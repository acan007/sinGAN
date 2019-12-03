import math
import os
import sys

import click
import torch
import matplotlib.pyplot as plt
from torch import nn

from chanye._utils_torch import show_torch, show_batch_torch
from chanye._settings import set_dev_location, get_dataset_path, set_numpy_precision, seed_random
from utils import get_config
from model import SinGAN


class SuperResolution(SinGAN):
    def __init__(self, config, dataset_path):
        super().__init__(config, dataset_path)

    def adjust_scale_factor_by_image(self):
        scale_factor = math.pow(1 / 2, 1 / 3)
        num_scale = round(math.log(1 / self.config['sr_factor'], scale_factor))
        scale_factor = pow(self.config['sr_factor'], 1 / num_scale)

        self.config['num_scale'] = num_scale
        self.config['scale_factor'] = scale_factor

    def test_samples(self, save):
        os.makedirs(self.path_sample, exist_ok=True)

        self.eval_mode_all()
        with torch.no_grad():
            _, _, width, height = self.real_pyramid[-1].shape
            generator_0 = self.generator_pyramid[-1]
            super_resolution = self.real_pyramid[-1]
            self.repo = []
            for scale in range(self.num_scale):
                width_sr = round(width * self.scale_factor ** scale)
                height_sr = round(height * self.scale_factor ** scale)

                if scale:
                    super_resolution = nn.Upsample((width_sr, height_sr))(super_resolution)
                noise_optimal = torch.zeros_like(super_resolution)

                super_resolution = generator_0(super_resolution, noise_optimal)
                self.repo.append(super_resolution)

            super_resolution = show_torch(super_resolution.clamp(-1, 1), return_img=True)

            if save:
                save_name = os.path.join(self.path_sample, "super_resolution")
                plt.imsave(save_name, super_resolution)
                print("Result Saved:" + save_name)
        return super_resolution


@click.command()
@click.option('--config', type=str, default='./config/SR.yaml', help='Path to the config file.')
@click.option('--dataset_path', type=str, default='./assets/Input', help='Path to root dataset path')
@click.option('--resume', default=False, help='whether resume and train')
def main(config, dataset_path, resume):
    seed_random()
    set_numpy_precision()

    # model
    model = SuperResolution(get_config(config), dataset_path)
    # TODO
    # if resume:
    #     model.load_models(load_path, resume)

    # train
    model.train()


if __name__ == '__main__':
    main()
