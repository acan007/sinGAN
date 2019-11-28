import cv2
import torch
import os
import numpy as np
from torch import nn

from _utils_torch import reset_gradients, reshape_batch_torch
from _visualizer import denormalize
from _utils_date import get_todate

from networks import Generator, Discriminator
from utils import get_scheduler, weights_init


class SinGAN(nn.Module):
    def __init__(self, input, config, device):
        super(SinGAN, self).__init__()

        self.config = config
        self.device = device

        self.lr = config['lr']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.weight_decay = config['weight_decay']
        num_scale = config['num_scale']
        scale_factor = config['scale_factor']

        self.one = torch.FloatTensor([1])[0]
        self.mone = torch.FloatTensor([-1])[0]  # self.one * -1
        self.w_recon = config['w_recon']
        self.w_gp = config['w_gp']

        width, height, _ = input.shape
        self.height_pyramid, self.width_pyramid, self.real_pyramid = [], [], []
        for i in range(num_scale, -1, -1):
            multiplier = (1 / scale_factor) ** i

            height_scaled = int(round(height * multiplier))
            width_scaled = int(round(width * multiplier))

            self.height_pyramid.append(height_scaled)
            self.width_pyramid.append(width_scaled)

            raw_scaled = cv2.resize(input, (height_scaled, width_scaled))
            raw_scaled = torch.tensor(np.transpose(raw_scaled, [2, 0, 1])[np.newaxis])
            self.real_pyramid.append(raw_scaled)

        self.noise_optimal_pyramid = []
        for scale in range(num_scale):
            if not scale:
                noise = torch.randn_like(self.real_pyramid[scale])
            else:
                noise = torch.zeros_like(self.real_pyramid[scale])
            self.noise_optimal_pyramid.append(noise)

        self.generator_pyramid, self.discriminator_pyramid = [], []
        for scale in range(num_scale):
            dim = 32 * 2 ** (scale // 4)  # increase dim by a factor of 2 every 4 scales
            self.generator_pyramid.append(Generator(input_dim=3, dim=dim, n_layers=5))
            self.discriminator_pyramid.append(Discriminator(input_dim=3, dim=dim, n_layers=5))

        # TODO
        # params_d = list(self.discriminator_a.parameters()) + list(self.discriminator_b.parameters())
        # params_g = list(self.generator_a.parameters()) + list(self.generator_b.parameters())
        # self.optimizer_d = torch.optim.Adam(params_d, lr, (beta1, beta2), weight_decay=weight_decay)
        # self.optimizer_g = torch.optim.Adam(params_g, lr, (beta1, beta2), weight_decay=weight_decay)

        # self.scheduler_d = get_scheduler(self.optimizer_d, config)
        # self.scheduler_g = get_scheduler(self.optimizer_g, config)
        #
        # self.apply(weights_init(config['init']))
        # self.discriminator_a.apply(weights_init('gaussian'))
        # self.discriminator_b.apply(weights_init('gaussian'))
        #
        self.criterion_l1 = nn.L1Loss()
        # self.criterion_l2 = nn.MSELoss()

        self.to(device)

    def update_scheduler(self):
        if self.scheduler_d and self.scheduler_g:
            self.scheduler_d.step()
            self.scheduler_g.step()

    def generate_fake_image(self, scale):  # TODO : naming
        generated = None
        for s in range(scale + 1):
            if type(generated) != type(None):
                generated = nn.Upsample((self.width_pyramid[s], self.height_pyramid[s]))(generated)

            noise = torch.randn_like(self.real_pyramid[s])
            generator = self.generator_pyramid[s]

            generated = generator(generated, noise)

        return generated

    def update_d(self, real, fake):
        reset_gradients([self.optimizer_d, self.optimizer_g])

        logit_real = self.discriminator(real)
        self.loss_d_real = logit_real.mean()
        self.loss_d_real.backward(self.mone)

        logit_fake = self.discriminator(fake.detach())
        self.loss_d_fake = logit_fake.mean()
        self.loss_d_fake.backward(self.one)

        self.gradient_penalty = self.discriminator.calculate_gradient_penalty(real, fake) * self.w_gp
        self.gradient_penalty.backward()

        self.optimizer_d.step()

    def update_g(self, real, fake, noise_optimal):
        reset_gradients([self.optimizer_d, self.optimizer_g])

        # TODO : real을 upsampled 된 애로 바꾸자
        self.loss_recon = self.criterion_l1(self.generator(real, noise_optimal), real) * self.w_recon
        self.loss_recon.backward()

        logit_g = self.discriminator(fake)
        self.loss_g = logit_g.mean()
        self.loss_g.backward(self.mone, )

        self.optimizer_g.step()

    def train_pyramid(self, scale):
        self.scale = scale

        self.real = self.real_pyramid[self.scale]
        self.generator = self.generator_pyramid[self.scale]
        self.discriminator = self.discriminator_pyramid[self.scale]
        self.noise_optimal = self.noise_optimal_pyramid[self.scale]
        # self.recon_img_upsampled = se

        params_d = list(self.discriminator.parameters())
        params_g = list(self.generator.parameters())
        self.optimizer_d = torch.optim.Adam(params_d, self.lr, (self.beta1, self.beta2), weight_decay=self.weight_decay)
        self.optimizer_g = torch.optim.Adam(params_g, self.lr, (self.beta1, self.beta2), weight_decay=self.weight_decay)

        # self.scheduler_d = get_scheduler(self.optimizer_d, self.config)
        # self.scheduler_g = get_scheduler(self.optimizer_g, self.config)

        # --- init weights
        if not scale % 4:
            self.apply(weights_init(self.config['init']))
            self.discriminator.apply(weights_init('gaussian'))
        else:
            print("Copy weight from previous pyramid G & D")
            self.generator.load_state_dict(self.generator_pyramid[scale - 1].state_dict())
            self.discriminator.load_state_dict(self.discriminator_pyramid[scale - 1].state_dict())

        # --- forward and back prob
        for step in range(self.config['d_step']):
            self.fake = self.generate_fake_image(scale)
            self.update_d(self.real, self.fake)

        for step in range(self.config['g_step']):
            self.fake = self.generate_fake_image(scale)
            self.update_g(self.real, self.fake, self.noise_optimal)

    def eval_mode_all(self):
        self.generator_a.eval()
        self.generator_b.eval()
        self.discriminator_a.eval()
        self.discriminator_b.eval()

    def print_log(self, scale, step):
        loss_d_real = self.loss_d_real.item()
        loss_d_fake = self.loss_d_fake.item()
        loss_gp = self.gradient_penalty.item()

        loss_g = self.loss_recon.item()
        loss_recon = self.loss_g.item()

        print(
            '[{} scale / {} step] Dis - real: {:4.2}, fake: {:4.2}, gp: {:4.2} / Gen - adv: {:4.2}, recon: {:4.2}'. \
                format(scale, step, loss_d_real, loss_d_fake, loss_gp, loss_g, loss_recon)
        )

    def save_models(self, current_step, total_step, folder_name='models', save_name='dummy'):
        filename = os.path.join(folder_name, save_name)
        os.makedirs(folder_name, exist_ok=True)

        state = {
            'dis_a': self.discriminator_a.state_dict(),
            'dis_b': self.discriminator_b.state_dict(),
            'gen_a': self.generator_a.state_dict(),
            'gen_b': self.generator_b.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'current_step': current_step,
            'total_step': total_step,
        }

        torch.save(state, filename)

    def load_models(self, model_path, train=True):
        checkpoint = torch.load(model_path)

        # weight
        if train:
            self.discriminator_a.load_state_dict(checkpoint['dis_a'])
            self.discriminator_b.load_state_dict(checkpoint['dis_b'])
        self.generator_a.load_state_dict(checkpoint['gen_a'])
        self.generator_b.load_state_dict(checkpoint['gen_b'])

        # optimizer
        if train:
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        return checkpoint['current_step'], checkpoint['total_step']

    def test_samples(self, test_a, test_b, n_test_style):
        self.eval_mode_all()
        with torch.no_grad():
            repo = []
            style_random = torch.randn(n_test_style, self.config['gen']['style_dim']).to(self.device)
            for i in range(test_a.size(0)):
                test_a_i = test_a[i].unsqueeze_(0)
                test_b_i = test_b[i].unsqueeze_(0)

                style_a, content_a = self.generator_a.encode(test_a_i)
                style_b, content_b = self.generator_b.encode(test_b_i)

                content_a_tile = content_a.repeat(n_test_style, 1, 1, 1)
                content_b_tile = content_b.repeat(n_test_style, 1, 1, 1)

                fake_a = self.generator_a.decode(style_random, content_b_tile)
                recon_a = self.generator_a.decode(style_a, content_a)

                fake_b = self.generator_b.decode(style_random, content_a_tile)
                recon_b = self.generator_b.decode(style_b, content_b)

                repo.append(torch.cat([test_a_i, fake_b, recon_a, test_b_i, fake_a, recon_b]))

            image_saving = denormalize(reshape_batch_torch(torch.cat(repo, 2), -1, 1)[0])
            return image_saving
