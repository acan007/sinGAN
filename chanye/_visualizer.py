import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


# jpg : 0 ~ 255, # png: 0 ~ 1
def normalize(img, mode='tanh'):
    assert not (np.max(img) > 1.5 and np.min(img) < -1.5)
    if np.max(img) < 1.5:  # png
        img = img * 255.

    if mode == 'tanh':
        img = np.array(img) / 127.5 - 1
    elif mode == 'sigmoid':
        img = np.array(img) / 255
    else:
        img = None
    return img


def denormalize(img, mode='tanh'):
    assert np.max(img) < 1.5 and np.min(img) > -1.5
    if mode == 'tanh':
        img = ((np.array(img) + 1) * 255 / 2).astype(np.uint8)
    elif mode == 'sigmoid':
        img = (np.array(img) * 255).astype(np.uint8)
    else:
        img = None
    return img


def load_img(path):
    image = plt.imread(path)
    if len(image.shape) < 3:
        image = image[..., np.newaxis]
    return image


def preprocess(img):  # shape, dimension 맞춰주기, denormalize
    img = img[..., :3]  # png format has 4 channels
    if np.max(img) > 10:
        img = (img / 255).astype(np.float32)
    if np.min(img) < 0:
        img = denormalize(img, 'tanh')
    if len(img.shape) == 4:
        img = img[0]
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    return img


def reshape_batch(images, n_cols=None, n_rows=None, padding=0):
    if len(images.shape) < 4 and images.shape[0] > 1:
        images = images[..., np.newaxis]
    batch_size, h, w, c = images.shape

    if not n_rows and not n_cols:
        # n_cols = batch_size
        # n_rows = 1

        n_cols = 5
        n_dummy = n_cols - batch_size % n_cols
        dummy = np.zeros_like(images[:n_dummy])
        images = np.concatenate([images, dummy])

        batch_size, h, w, c = images.shape
        n_rows = batch_size // n_cols

    elif n_rows == -1 or n_cols == -1:
        n_cols = batch_size // n_rows if n_cols == -1 else n_cols
        n_rows = batch_size // n_cols if n_rows == -1 else n_rows

    if batch_size % n_rows:
        n_dummy = n_cols - batch_size % n_cols
        dummy = np.zeros_like(images[0])
        images += [dummy] * n_dummy

    if padding:
        pad_size = ((0, 0), (padding, padding), (padding, padding), (0, 0))
        images = np.pad(images, pad_size, 'constant', constant_values=1)
        h += padding * 2
        w += padding * 2

    images = np.reshape(images, [n_rows, n_cols, h, w, c])
    images = np.transpose(images, [0, 2, 1, 3, 4])
    images = np.reshape(images, [n_rows * h, n_cols * w, c])
    return images, n_cols, n_rows


def show(img, figsize=None, show_ticks=False, return_img=False, *a, **k):
    img = preprocess(img)

    if figsize:
        plt.figure(figsize=figsize)

    plt.imshow(img, *a, **k)
    if not show_ticks:
        plt.xticks([])
        plt.yticks([])
    plt.show()
    if return_img:
        return img


def show_batch(images, n_cols=None, n_rows=None, padding=0, figsize=None, *a, **k):
    images, n_cols, n_rows = reshape_batch(images, n_cols, n_rows, padding)
    figsize = figsize if figsize else (n_cols * 4, n_rows * 4)
    return show(images, figsize=figsize, *a, **k)


def show_multiple(image_list, n_cols, n_rows, show_ticks=False, figsize=(10, 10)):
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize)
    for n, ax in enumerate(axes.flatten()):
        img = preprocess(image_list[n])
        ax.imshow(img)
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


def show_scatter(img, dots, ax=False, figsize=None, scatter_size=15):
    img = preprocess(img)
    ax = ax if ax else plt

    if figsize:
        plt.figure(figsize=figsize)
    ax.imshow(img)
    if type(dots) == list:
        [ax.scatter(i[:, 0], i[:, 1], s=scatter_size) for i in dots]
    else:
        ax.scatter(dots[:, 0], dots[:, 1], s=scatter_size)


def show_scatter_batch(img_batch, dots_batch, ncols=False, nrows=False, k=1.5):
    if not nrows and not ncols:
        nrows = 1
        ncols = img_batch.shape[0]

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols * k, 4 * nrows * k))
    for i in range(nrows * ncols):
        img = img_batch[i]
        dots = dots_batch[i]
        ax = axes[i // ncols][i % ncols] if nrows != 1 else axes[i]
        show_scatter(img, dots, ax=ax)
    fig.tight_layout()


def clear_jupyter_console():
    clear_output()
