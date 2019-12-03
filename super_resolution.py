import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from chanye._utils_torch import show_torch, show_batch_torch

from model import SinGAN


class SuperResolution(SinGAN):
    def __init__(self, input, config, device):
        super().__init__(input, config, device)

    def super_resolution(self, save):
        os.makedirs(self.path_sample, exist_ok=True)

        self.eval_mode_all()
        with torch.no_grad():
            _, _, width, height = self.real_pyramid[-1].shape
            generator_0 = self.generator_pyramid[-1]
            super_resolution = self.real_pyramid[-1]
            self.repo = []
            for scale in range(self.config['num_scale']):
                width_sr = round(width * self.config['scale_factor'] ** scale)
                height_sr = round(height * self.config['scale_factor'] ** scale)

                if scale:
                    super_resolution = nn.Upsample((width_sr, height_sr))(super_resolution)
                noise_optimal = torch.zeros_like(super_resolution)

                super_resolution = generator_0(super_resolution, noise_optimal)
                self.repo.append(super_resolution)

            super_resolution = show_torch(super_resolution.clamp(-1, 1), return_img=True)

            if save:
                save_name = os.path.join(self.path_sample, "super_resolution")
                plt.imsave(save_name, super_resolution)
        return super_resolution
