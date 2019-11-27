import numpy as np
import os
import click
import torch
import matplotlib.pyplot as plt

from _visualizer import clear_jupyter_console, save_images
from _settings import set_dev_location, get_dataset_path, set_numpy_precision
from _telegramer import send_text
from _utils_torch import *

from utils import get_config, adjust_scale_factor_by_image
from model import SinGAN
from data import UnpairedImageFileList, ImageFolder, get_data_loader


def loader_edges2shoes(config, is_test_iter=False):
    train_path = os.path.join(get_dataset_path(), config['path_data'], 'train')
    train_loader, dataset = get_data_loader(config, ImageFolder, True, root=train_path)

    test_path = os.path.join(get_dataset_path(), config['path_data'], 'val')
    test_loader, _ = get_data_loader(config, ImageFolder, False, root=test_path)

    if is_test_iter:
        test_iter = iter(test_loader)
        return train_loader, test_iter, dataset
    return train_loader, test_loader, dataset


def loader_celebA(config):
    base_path = os.path.join(get_dataset_path(), config['path_data'])
    return get_data_loader(config, UnpairedImageFileList, True, dataroot=base_path,
                           file_a='./datasets/anno_male', file_b='./datasets/anno_female')


@click.command()
@click.option('--config', type=str, default='./config/random_sample.yaml', help='Path to the config file.')
@click.option('--location', type=str, default='macbook', help='dev env [macbook | server | home]')
@click.option('--load_path', default='', help='path for saved model')
@click.option('--resume', default=False, help='whether resume and train')
def main(config, location, load_path, resume):
    set_dev_location(location)
    set_numpy_precision()

    config = get_config(config)

    # data loader
    path_img_load = os.path.join(config['path_data_base'], config['path_data'])
    path_img_save = config['path_img_save']
    path_model_save = config['path_model_save']
    input = plt.imread(path_img_load)[:, :, :3]
    config = adjust_scale_factor_by_image(input, config)

    # train opts
    image_display_iter = config['image_display_iter']
    image_save_iter = config['image_save_iter']
    model_save_iter = config['model_save_iter']

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SinGAN(input, config, device)
    # TODO
    # if load_path:
    #     model.load_models(load_path, resume)

    # train
    print("Start sinGAN Training")
    for scale in range(config['num_scale']):
        for step in range(config['n_iter']):
            model.train_pyramid(scale)
            if not (step + 1) % 100:
                model.print_log(scale, step + 1)
        print(scale, "-" * 100)
    #
    #
    # iters = 0
    # while iters <= max_iters:
    #     for iters, (real_a, real_b, _, _) in enumerate(train_loader):
    #         model.update_scheduler()
    #
    #         real_a = real_a.to(device)
    #         real_b = real_b.to(device)
    #
    #         model.forward(real_a, real_b)
    #
    #         model.update_g()
    #         model.update_d()
    #         if device.type == 'cuda':
    #             torch.cuda.synchronize()
    #
    #         # logger
    #         if not (iters + 1) % image_display_iter:
    #             model.print_log(iters, max_iters)
    #             show_batch_torch(torch.cat([real_a, model.fake_b, model.recon_a, real_b, model.fake_a, model.recon_b]))
    #
    #         # save image
    #         if not (iters + 1) % image_save_iter:
    #             clear_jupyter_console()
    #
    #             img_saving = model.test_samples(real_a, real_b, n_test_style)
    #             img_saving = np.concatenate([img_saving[:, :save_width, :], img_saving[:, save_width:, :]], axis=0)
    #             save_images(img_saving, path_img_save, '{:03}.png'.format(iters + 1))
    #
    #         # save model
    #         if not (iters + 1) % model_save_iter:
    #             model.save_models(iters, max_iters, path_model_save, '{:02}'.format((iters + 1) // model_save_iter))
    #
    # send_text('MUNIT LEARNING FINISHED')


if __name__ == '__main__':
    main()
