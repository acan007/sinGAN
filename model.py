import math

import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from chanye._utils_torch import reset_gradients, reshape_batch_torch

from networks import Generator, Discriminator
from utils import get_scheduler, weights_init, normalize_image


class SinGAN(nn.Module):
    def __init__(self, config, dataset_path):
        super(SinGAN, self).__init__()

        # --- member variables
        self.config = config
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input = plt.imread(os.path.join(dataset_path, config['path_data']))
        self.input = normalize_image(input)
        self.adjust_scale_factor_by_image()

        self.lr = config['lr']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.weight_decay = config['weight_decay']
        self.num_scale = self.config['num_scale']
        self.scale_factor = self.config['scale_factor']

        self.name = config['path_data'].split('/')[-1].split('.')[0] + "_" + config['mode']
        self.path_model = os.path.join(config['path_model_save'], self.name)
        self.path_sample = os.path.join(config['path_sample_save'], self.name)

        self.one = torch.FloatTensor([1])[0].to(self.device)
        self.mone = torch.FloatTensor([-1])[0].to(self.device)  # self.one * -1
        self.w_recon = self.config['w_recon']
        self.w_gp = self.config['w_gp']

        # --- pyramids
        width, height, _ = self.input.shape
        self.height_pyramid, self.width_pyramid, self.real_pyramid = [], [], []
        for scale in range(self.num_scale - 1, -1, -1):
            multiplier = (1 / self.scale_factor) ** scale

            height_scaled = int(round(height * multiplier))
            width_scaled = int(round(width * multiplier))

            self.height_pyramid.append(height_scaled)
            self.width_pyramid.append(width_scaled)

            processed = cv2.resize(self.input, (height_scaled, width_scaled))
            processed = torch.tensor(np.transpose(processed, [2, 0, 1])[np.newaxis])
            self.real_pyramid.append(processed.to(self.device, torch.float))

        self.noise_optimal_pyramid = []
        for scale in range(self.num_scale):
            if not scale:
                noise = torch.randn_like(self.real_pyramid[scale]).to(self.device)
            else:
                noise = torch.zeros_like(self.real_pyramid[scale]).to(self.device)
            self.noise_optimal_pyramid.append(noise)

        self.generator_pyramid, self.discriminator_pyramid = [], []
        for scale in range(self.num_scale):
            dim = 32 * 2 ** (scale // 4)  # increase dim by a factor of 2 every 4 scales
            if self.device.type == 'cuda':
                self.generator_pyramid.append(Generator(input_dim=3, dim=dim, n_layers=5).cuda())
                self.discriminator_pyramid.append(Discriminator(input_dim=3, dim=dim, n_layers=5).cuda())
            else:
                self.generator_pyramid.append(Generator(input_dim=3, dim=dim, n_layers=5))
                self.discriminator_pyramid.append(Discriminator(input_dim=3, dim=dim, n_layers=5))

        self.sigma_pyramid = []
        self.assign_network_at_scale_begin(0)

        self.criterion_l2 = nn.MSELoss()
        self.to(device)

    def adjust_scale_factor_by_image(self):
        img_shape = self.input.shape
        idx_dim = np.argsort(img_shape)[::-1][:2]
        min_dim = min(img_shape[idx_dim[0]], img_shape[idx_dim[1]])

        num_scale = int(np.ceil(np.log(min_dim / self.config['coarsest_dim']) / np.log(self.config['scale_factor_init'])))
        scale_factor = np.power(min_dim / self.config['coarsest_dim'], 1 / num_scale)

        self.config['num_scale'] = num_scale
        self.config['scale_factor'] = scale_factor

    def get_sigma(self, scale, real):
        if not scale:
            sigma = 1

        else:
            recon = None
            for s in range(scale):
                generator = self.generator_pyramid[s]
                noise_optimal = self.noise_optimal_pyramid[s]
                sigma = self.sigma_pyramid[s]

                recon = generator(recon, noise_optimal * sigma)
                recon = nn.Upsample((self.width_pyramid[s + 1], self.height_pyramid[s + 1]))(recon)

            sigma = torch.sqrt(self.criterion_l2(real, recon))

        self.sigma_pyramid.append(sigma)
        assert len(self.sigma_pyramid) == (scale + 1)
        return sigma

    def generate_fake_image(self, scale):
        if scale == -1:
            scale = self.num_scale - 1

        fake_image = None
        for s in range(scale + 1):
            if type(fake_image) != type(None):
                fake_image = nn.Upsample((self.width_pyramid[s], self.height_pyramid[s]))(fake_image)

            generator = self.generator_pyramid[s]
            noise = torch.randn_like(self.real_pyramid[s]).to(self.device)
            sigma = self.sigma_pyramid[s]

            fake_image = generator(fake_image, noise * sigma)

        return fake_image

    def generate_recon_image(self, scale):
        if scale == -1:
            scale = self.num_scale - 1

        recon_image = None
        for s in range(scale + 1):
            if type(recon_image) != type(None):
                recon_image = nn.Upsample((self.width_pyramid[s], self.height_pyramid[s]))(recon_image)

            generator = self.generator_pyramid[s]
            noise_optimal = self.noise_optimal_pyramid[s]
            sigma = self.sigma_pyramid[s]

            recon_image = generator(recon_image, noise_optimal * sigma)

        return recon_image

    def assign_network_at_scale_begin(self, scale):
        self.generator = self.generator_pyramid[scale]
        self.discriminator = self.discriminator_pyramid[scale]
        self.noise_optimal = self.noise_optimal_pyramid[scale]

        params_d = list(self.discriminator.parameters())
        params_g = list(self.generator.parameters())
        self.optimizer_d = torch.optim.Adam(params_d, self.lr, (self.beta1, self.beta2),
                                            weight_decay=self.weight_decay)
        self.optimizer_g = torch.optim.Adam(params_g, self.lr, (self.beta1, self.beta2),
                                            weight_decay=self.weight_decay)

        # --- init weights
        if not scale % 4:
            print("Init weight G & D")
            self.apply(weights_init(self.config['init']))
            self.discriminator.apply(weights_init('gaussian'))
        else:
            print("Copy weight from previous pyramid G & D")
            self.generator.load_state_dict(self.generator_pyramid[scale - 1].state_dict())
            self.discriminator.load_state_dict(self.discriminator_pyramid[scale - 1].state_dict())

    def assign_parameters_at_scale_begin(self, scale):
        self.scale = scale
        self.real = self.real_pyramid[scale]
        self.sigma = self.get_sigma(scale, self.real)

    def update_d(self):
        reset_gradients([self.optimizer_d, self.optimizer_g])

        logit_real = self.discriminator(self.real)
        self.loss_d_real = logit_real.mean()
        self.loss_d_real.backward(self.mone)

        logit_fake = self.discriminator(self.fake.detach())
        self.loss_d_fake = logit_fake.mean()
        self.loss_d_fake.backward(self.one)

        self.gradient_penalty = self.discriminator.calculate_gradient_penalty(self.real, self.fake,
                                                                              self.device) * self.w_gp
        self.gradient_penalty.backward()

        self.optimizer_d.step()

    def update_g(self):
        reset_gradients([self.optimizer_d, self.optimizer_g])

        self.loss_recon = self.criterion_l2(self.recon, self.real) * self.w_recon
        self.loss_recon.backward()

        logit_g = self.discriminator(self.fake)
        self.loss_g = logit_g.mean()
        self.loss_g.backward(self.mone)

        self.optimizer_g.step()

    def train_single_scale_pyramid(self, scale):
        # self.scheduler_d = get_scheduler(self.optimizer_d, self.config)
        # self.scheduler_g = get_scheduler(self.optimizer_g, self.config)

        # --- forward and back prob
        for step in range(self.config['d_step']):
            self.fake = self.generate_fake_image(scale)
            self.update_d()

        for step in range(self.config['g_step']):
            self.fake = self.generate_fake_image(scale)
            self.recon = self.generate_recon_image(scale)
            self.update_g()

    def train(self):
        print("Start sinGAN Training - {}, {} scales".format(self.name, self.num_scale))
        for scale in range(self.num_scale):
            for step in range(self.config['n_iter']):
                if not step:
                    print("scale", scale + 1, "-" * 100)
                    self.assign_network_at_scale_begin(scale)
                    self.assign_parameters_at_scale_begin(scale)

                self.train_single_scale_pyramid(scale)
                if not (step + 1) % self.config['log_iter']:
                    self.print_log(scale + 1, step + 1)

            self.save_image = self.test_samples_scale(scale, save=True)
            self.save_models(scale)

        print("sinGAN Training Finished")

    def eval_mode_all(self):
        self.generator.eval()
        self.discriminator.eval()

    def print_log(self, scale, step):
        loss_d_real = self.loss_d_real.item()
        loss_d_fake = self.loss_d_fake.item()
        loss_gp = self.gradient_penalty.item()

        loss_g = self.loss_g.item()
        loss_recon = self.loss_recon.item()

        print(
            '[{} scale / {} step] Dis - real: {:4.2}, fake: {:4.2}, gp: {:4.2} / Gen - adv: {:4.2}, recon: {:4.2}'. \
                format(scale, step, loss_d_real, loss_d_fake, loss_gp, loss_g, loss_recon)
        )

    def save_models(self, scale):
        os.makedirs(self.path_model, exist_ok=True)

        state = {
            'generator_pyramid': self.generator_pyramid,
            'discriminator_pyramid': self.discriminator_pyramid,
            'noise_optimal_pyramid': self.noise_optimal_pyramid,
            'sigma_pyramid': self.sigma_pyramid,
            'real_pyramid': self.real_pyramid,
            'current_scale': scale,
            'current_optimizer_d': self.optimizer_d.state_dict(),
            'current_optimizer_g': self.optimizer_g.state_dict(),
        }

        save_name = os.path.join(self.path_model, "scale_{:02}".format(scale))
        torch.save(state, save_name)

    def load_models(self, scale, train=True):
        if scale == -1:
            scale = self.num_scale - 1

        save_name = os.path.join(self.path_model, "scale_{:02}".format(scale))
        checkpoint = torch.load(save_name)

        # weight
        self.generator_pyramid = checkpoint['generator_pyramid']
        self.discriminator_pyramid = checkpoint['discriminator_pyramid']
        self.noise_optimal_pyramid = checkpoint['noise_optimal_pyramid']
        self.sigma_pyramid = checkpoint['sigma_pyramid']
        self.real_pyramid = checkpoint['real_pyramid']
        self.scale = checkpoint['current_scale']

        if train:
            self.optimizer_d.load_state_dict(checkpoint['current_optimizer_d'])
            self.optimizer_g.load_state_dict(checkpoint['current_optimizer_g'])

    def test_samples_scale(self, scale, save):
        os.makedirs(self.path_sample, exist_ok=True)

        self.eval_mode_all()
        with torch.no_grad():
            save_image = reshape_batch_torch(
                torch.cat([self.generate_fake_image(scale).clamp(-1, 1) for _ in range(20)]),
                n_cols=4, n_rows=5, padding=2
            )
            if save:
                save_name = os.path.join(self.path_sample, "scale_{:02}".format(scale))
                plt.imsave(save_name, save_image)
                print("Result Saved:" + save_name)
        return save_image

    def test_samples(self, save):
        return self.test_samples_scale(-1, save)