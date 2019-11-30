import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from _utils_torch import show_torch

from model import SinGAN


class Paint2Image(SinGAN):
    def __init__(self, naive_img, input, config, device):
        super().__init__(input, config, device)

        if np.max(naive_img) < 10:
            naive_img = naive_img * 255
        self.naive_img = naive_img / 127.5 - 1

        self.naive_pyramid = []
        for i in range(self.config['num_scale']):
            height_scaled = self.height_pyramid[i]
            width_scaled = self.width_pyramid[i]

            processed = cv2.resize(naive_img, (height_scaled, width_scaled))
            processed = torch.tensor(np.transpose(processed, [2, 0, 1])[np.newaxis])
            self.naive_pyramid.append(processed.to(self.device, torch.float))

    # def generate_fake_image(self, scale):

    def generate_paint2image(self, init_scale):
        repo = []
        for scale in range(init_scale, self.config['num_scale']):  # TODO : check "+1"
            generator = self.generator_pyramid[scale]
            noise_optimal = self.noise_optimal_pyramid[scale]
            sigma = self.sigma_pyramid[scale]

            if scale == init_scale:
                naive = self.naive_pyramid[init_scale]
                recon_image = generator(naive, noise_optimal * sigma)

            else:
                recon_image = nn.Upsample((self.width_pyramid[scale], self.height_pyramid[scale]))(recon_image)
                recon_image = generator(recon_image, noise_optimal * sigma)

            repo.append(recon_image)
        return recon_image

    def test_samples(self, init_scale, save):
        os.makedirs(self.path_sample, exist_ok=True)

        self.eval_mode_all()
        with torch.no_grad():
            save_image = show_torch(self.generate_paint2image(init_scale).clamp(-1, 1))
            if save:
                save_name = os.path.join(self.path_sample, "scale_{:02}".format(self.scale))
                plt.imsave(save_name, save_image)
        return save_image
