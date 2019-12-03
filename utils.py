from torch.nn import init
from torch.optim import lr_scheduler

import numpy as np
import yaml
import math


def get_config(path):
    with open(path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


def normalize_image(image):
    if np.max(image) < 10:
        image = image[:, :, :3] * 255
    return image / 127.5 - 1


def seed_random():
    import random
    # import tensorflow as tf
    import torch

    random.seed(123)
    np.random.seed(123)
    # tf.set_random_seed(123)
    torch.manual_seed(123)


def set_numpy_precision(precision=4):
    np.set_printoptions(precision=precision, linewidth=200, edgeitems=5, suppress=True)


def get_scheduler(optimizer, config, iterations=-1):
    policy = config.get('lr_policy', None)
    if not policy or policy == 'constant':
        scheduler = None  # constant scheduler
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['step_size'],
                                        gamma=config['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
