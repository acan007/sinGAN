import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable

from _layers import LinearBlock, Conv2dBlock, ResBlock

from utils import get_config


class Generator(nn.Module):
    def __init__(self, input_dim, dim, n_layers):
        super(Generator, self).__init__()

        self.pad = nn.ZeroPad2d(int(n_layers))

        layers = []
        layers += [Conv2dBlock(input_dim, dim, kernel_size=3, stride=1, norm='bn', activation='lrelu')]
        for n in range(n_layers - 2):
            layers += [Conv2dBlock(dim, dim, kernel_size=3, stride=1, norm='bn', activation='lrelu')]

        layers += [Conv2dBlock(dim, 3, kernel_size=3, stride=1, activation='tanh')]

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, noise):
        x_pad = self.pad(x)
        noise_pad = self.pad(x)
        # logit = x + noise
        # for layer in self.layers:
        #     logit = layer(logit)
        #     print(logit.shape)
        # return x[:, :, 5:-5, 5:-5] + logit
        return x + self.model(x_pad + noise_pad)


class Discriminator(nn.Module):
    def __init__(self, input_dim, dim, n_layers):
        super(Discriminator, self).__init__()

        self.pad = nn.ZeroPad2d(int(n_layers))

        layers = []
        layers += [Conv2dBlock(input_dim, dim, kernel_size=3, stride=1, norm='bn', activation='lrelu')]
        for n in range(n_layers - 2):
            layers += [Conv2dBlock(dim, dim, kernel_size=3, stride=1, norm='bn', activation='lrelu')]

        layers += [Conv2dBlock(dim, 1, kernel_size=3, stride=1)]

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(1, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(1, real_images.size(1), real_images.size(2), real_images.size(3))
        # if self.cuda:
        #     eta = eta.cuda(self.cuda_index)
        # else:
        #     eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # if self.cuda:
        #     interpolated = interpolated.cuda(self.cuda_index)
        # else:
        #     interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.forward(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  # grad_outputs=torch.ones(
                                  #     prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                  #     prob_interpolated.size()),
                                  grad_outputs=torch.ones(prob_interpolated.size()),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def forward(self, x):
        return self.model(self.pad(x))


if __name__ == '__main__':
    from utils import *
    import matplotlib.pyplot as plt

    raw = plt.imread('./Input/Images/birds.png')[:, :, :3]

    config = get_config('./config/random_sample.yaml')
    config = adjust_scale_factor_by_image(raw, config)

    generator_pyramid, discriminator_pyramid = [], []
    for scale in range(config['num_scale']):
        dim = 32 * 2 ** (scale // 4)  # increase dim by a factor of 2 every 4 scales
        generator_pyramid.append(Generator(input_dim=3, dim=dim, n_layers=5))
        discriminator_pyramid.append(Discriminator(input_dim=3, dim=dim, n_layers=5))

    scale = 1
    generator = generator_pyramid[scale]
    discriminator = discriminator_pyramid[scale]

    real = raw
    real = torch.tensor(np.transpose(real, [2, 0, 1])[np.newaxis])
    noise = torch.rand_like(real)
    print('raw', real.shape, noise.shape)

    print(generator(real, noise).shape)
    print(discriminator(real).shape)
