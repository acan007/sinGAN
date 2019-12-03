import numpy as np
import os


def set_dev_location(key):
    os.environ['dev_location'] = key


# methods for get proper dataset_path for different environment
def get_dataset_path():
    if os.environ['dev_location'] == 'server':
        return "/mnt/disks/sdb/datasets/"
    elif os.environ['dev_location'] == 'macbook':
        return "/Users/bochan/_datasets/"
    elif os.environ['dev_location'] == 'home':
        return "./datasets/"
    raise ValueError


def get_telegram_token_path():  # 'macbook', 'desktop', server1, 2, 3...
    if os.environ['dev_location'] == 'server':
        return '/home/bochan/_chanye/tokens/telegram.txt'
    elif os.environ['dev_location'] == 'macbook':
        return '/Users/bochan/_chanye/tokens/telegram.txt'
    else:
        raise ValueError


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


def jupyer_notebook_print_many():
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = 'all'


def use_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
