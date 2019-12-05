import math
import os
import sys
import click
import torch
import matplotlib.pyplot as plt
from torch import nn

from chanye._utils_torch import torch2numpy
from chanye._visualizer import preprocess
from sinGAN import SinGAN


class SuperResolution(SinGAN):
    def __init__(self, config):
        super().__init__(config)

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
            for scale in range(self.num_scale + 1):
                width_sr = round(width * self.scale_factor ** scale)
                height_sr = round(height * self.scale_factor ** scale)

                if scale:
                    super_resolution = nn.Upsample((width_sr, height_sr))(super_resolution)
                noise_optimal = torch.zeros_like(super_resolution)

                super_resolution = generator_0(super_resolution, noise_optimal)
                self.repo.append(super_resolution)

            super_resolution = preprocess(torch2numpy(super_resolution.clamp(-1, 1)))  # TODO : check

            if save:
                save_name = os.path.join(self.path_sample, "super_resolution.jpg")
                plt.imsave(save_name, super_resolution)
                print("Super resolution Saved:" + save_name)
        return super_resolution