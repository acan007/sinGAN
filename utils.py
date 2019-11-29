from torch.nn import init
from torch.optim import lr_scheduler

import numpy as np
import yaml
import math


def get_config(path):
    with open(path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


def adjust_scale_factor_by_image(img, config):
    idx_dim = np.argsort(img.shape)[::-1][:2]
    min_dim = min(img.shape[idx_dim[0]], img.shape[idx_dim[1]])

    num_scale = int(np.ceil(np.log(min_dim / config['coarsest_dim']) / np.log(config['scale_factor_init'])))
    scale_factor = np.power(min_dim / config['coarsest_dim'], 1 / num_scale)

    config['num_scale'] = num_scale
    config['scale_factor'] = scale_factor
    return config


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
