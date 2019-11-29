import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from _utils_torch import reset_gradients, reshape_batch_torch
from _visualizer import denormalize
from _utils_torch import show_batch_torch

from networks import Generator, Discriminator
from utils import get_scheduler, weights_init


class SinGAN(nn.Module):
    def __init__(self, input, config, device):
        super(SinGAN, self).__init__()

        if np.max(input) < 10:
            input * 255
        self.input = input / 127.5 - 1
        self.config = config
        self.device = device

        self.lr = config['lr']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.weight_decay = config['weight_decay']
        num_scale = config['num_scale']
        scale_factor = config['scale_factor']

        self.name = config['path_data'].split('/')[-1].split('.')[0] + "_" + config['mode']
        self.path_model = os.path.join(config['path_model_save'], self.name)
        self.path_sample = os.path.join(config['path_sample_save'], self.name)

        self.one = torch.FloatTensor([1])[0].to(self.device)
        self.mone = torch.FloatTensor([-1])[0].to(self.device)  # self.one * -1
        self.w_recon = config['w_recon']
        self.w_gp = config['w_gp']

        width, height, _ = self.input.shape
        self.height_pyramid, self.width_pyramid, self.real_pyramid = [], [], []
        for i in range(num_scale, -1, -1):
            multiplier = (1 / scale_factor) ** i

            height_scaled = int(round(height * multiplier))
            width_scaled = int(round(width * multiplier))

            self.height_pyramid.append(height_scaled)
            self.width_pyramid.append(width_scaled)

            processed = cv2.resize(self.input, (height_scaled, width_scaled))
            processed = torch.tensor(np.transpose(processed, [2, 0, 1])[np.newaxis])
            self.real_pyramid.append(processed.to(self.device, torch.float))
        #         self.input.to(device)

        self.noise_optimal_pyramid = []
        for scale in range(num_scale):
            if not scale:
                noise = torch.randn_like(self.real_pyramid[scale]).to(self.device)
            else:
                noise = torch.zeros_like(self.real_pyramid[scale]).to(self.device)
            self.noise_optimal_pyramid.append(noise)

        self.generator_pyramid, self.discriminator_pyramid = [], []
        for scale in range(num_scale):
            dim = 32 * 2 ** (scale // 4)  # increase dim by a factor of 2 every 4 scales
            self.generator_pyramid.append(Generator(input_dim=3, dim=dim, n_layers=5).cuda())
            self.discriminator_pyramid.append(Discriminator(input_dim=3, dim=dim, n_layers=5).cuda())

        self.sigma_pyramid = []

        self.assign_network_at_scale_begin(0)

        self.criterion_l2 = nn.MSELoss()

        self.to(device)

    def update_scheduler(self):
        if self.scheduler_d and self.scheduler_g:
            self.scheduler_d.step()
            self.scheduler_g.step()

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

    def generate_fake_image(self, scale):  # TODO : noise_amp
        if scale == -1:
            scale = self.config['num_scale']  # TODO : num_scale -1 인지 체크

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
        recon_image = None
        for s in range(scale + 1):
            if type(recon_image) != type(None):
                recon_image = nn.Upsample((self.width_pyramid[s], self.height_pyramid[s]))(recon_image)

            generator = self.generator_pyramid[s]
            noise_optimal = self.noise_optimal_pyramid[s]
            sigma = self.sigma_pyramid[s]

            recon_image = generator(recon_image, noise_optimal * sigma)

        return recon_image

    def update_d(self, real, fake):
        reset_gradients([self.optimizer_d, self.optimizer_g])

        logit_real = self.discriminator(real)
        self.loss_d_real = logit_real.mean()
        self.loss_d_real.backward(self.mone)

        logit_fake = self.discriminator(fake.detach())
        self.loss_d_fake = logit_fake.mean()
        self.loss_d_fake.backward(self.one)

        self.gradient_penalty = self.discriminator.calculate_gradient_penalty(real, fake, self.device) * self.w_gp
        self.gradient_penalty.backward()

        self.optimizer_d.step()

    def update_g(self, real, fake, recon):
        reset_gradients([self.optimizer_d, self.optimizer_g])

        # TODO : real을 upsampled 된 애로 바꾸자
        self.loss_recon = self.criterion_l2(recon, real) * self.w_recon
        self.loss_recon.backward()

        logit_g = self.discriminator(fake)
        self.loss_g = logit_g.mean()
        self.loss_g.backward(self.mone)

        self.optimizer_g.step()

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

    def train_single_scale_pyramid(self, scale):
        # self.scheduler_d = get_scheduler(self.optimizer_d, self.config)
        # self.scheduler_g = get_scheduler(self.optimizer_g, self.config)

        # --- forward and back prob
        for step in range(self.config['d_step']):
            self.fake = self.generate_fake_image(scale)
            self.update_d(self.real, self.fake)

        for step in range(self.config['g_step']):
            self.fake = self.generate_fake_image(scale)
            self.recon = self.generate_recon_image(scale)
            self.update_g(self.real, self.fake, self.recon)

    def train(self):
        print("Start sinGAN Training")
        for scale in range(self.config['num_scale']):
            for step in range(self.config['n_iter']):
                if not step:
                    print("scale", scale + 1, "-" * 100)
                    self.assign_network_at_scale_begin(scale)
                    self.assign_parameters_at_scale_begin(scale)

                self.train_single_scale_pyramid(scale)
                if not (step + 1) % self.config['log_iter']:
                    self.print_log(scale + 1, step + 1)

            self.save_image = self.test_samples(scale, True)
            self.save_models()

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

    def save_models(self):
        os.makedirs(self.path_model, exist_ok=True)

        state = {
            'generator_pyramid': self.generator_pyramid,
            'discriminator_pyramid': self.discriminator_pyramid,
            'noise_optimal_pyramid': self.noise_optimal_pyramid,
            'sigma_pyramid': self.sigma_pyramid,
            'real_pyramid': self.real_pyramid,
            'current_scale': self.scale,
            'current_optimizer_d': self.optimizer_d.state_dict(),
            'current_optimizer_g': self.optimizer_g.state_dict(),
        }

        save_name = os.path.join(self.path_model, "scale_{:02}".format(self.scale))
        torch.save(state, save_name)

    def load_models(self, scale, train=True):
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

    def test_samples(self, scale, save):
        os.makedirs(self.path_sample, exist_ok=True)

        self.eval_mode_all()
        with torch.no_grad():
            save_image = show_batch_torch(
                torch.cat([self.generate_fake_image(scale).clamp(-1, 1) for _ in range(20)]),
                n_cols=4, n_rows=5, padding=2
            )
            if save:
                save_name = os.path.join(self.path_sample, "scale_{:02}".format(scale))
                plt.imsave(save_name, save_image)
        return save_image
