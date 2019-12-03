import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable

from chanye._layers import Conv2dBlock


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
        # for coarsest scale
        if type(x) == type(None):
            noise_pad = self.pad(noise)
            return self.model(noise_pad)

        # else than coarsest scale
        else:
            x_pad = self.pad(x)
            noise_pad = self.pad(x)
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

    def calculate_gradient_penalty(self, real_images, fake_images, device):
        eta = torch.FloatTensor(1, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(1, real_images.size(1), real_images.size(2), real_images.size(3)).to(device)

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.forward(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def forward(self, x):
        return self.model(self.pad(x))
