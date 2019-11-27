import pandas as pd
import numpy as np
import os
import click
import matplotlib.pyplot as plt

from _settings import get_dataset_path, set_location_bochan
from _visualizer import show_batch


def write(root_path, name, flist):
    os.makedirs(root_path, exist_ok=True)
    path = os.path.join(root_path, name)
    with open(path, 'w') as f:
        for i in flist:
            f.write(i + '\n')


@click.command()
@click.option('--location', type=str, default='server', help='dev env [macbook | server]')
def main(location):
    set_location_bochan(location)

    attr_path = os.path.join(get_dataset_path(), 'celebA/Anno/list_attr_celeba.txt')
    with open(attr_path) as f:
        attr = f.readlines()

    columns = attr[1].strip().split(' ')
    df = pd.DataFrame([i.strip().replace('  ', ' ').split(' ') for i in attr[2:]])
    df.set_index(0, inplace=True)
    df = df.astype(np.int32)
    df.columns = columns

    base_path = os.path.join(get_dataset_path(), 'celebA/Img/img_align_celeba')

    idx_male = df[df.iloc[:, columns.index('Male')] == 1].index.tolist()
    idx_female = df[df.iloc[:, columns.index('Male')] == -1].index.tolist()

    images1 = np.array([plt.imread(os.path.join(base_path, i)) for i in np.random.choice(idx_male, 10)])
    images2 = np.array([plt.imread(os.path.join(base_path, i)) for i in np.random.choice(idx_female, 10)])

    show_batch(images1)
    show_batch(images2)

    write('./datasets/', 'anno_male', idx_male)
    write('./datasets/', 'anno_female', idx_female)


if __name__ == '__main__':
    main()
