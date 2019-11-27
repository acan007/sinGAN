from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import os.path
import random


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class UnpairedImageFileList(data.Dataset):
    def __init__(self, dataroot, config, file_a, file_b, transform=None, loader=default_loader):
        self.dataroot = dataroot
        self.config = config
        self.transform = transform
        self.loader = loader

        self.imlist_a = default_flist_reader(file_a)
        self.imlist_b = default_flist_reader(file_b)

        self.size_a = len(self.imlist_a)
        self.size_b = len(self.imlist_b)

        self.shuffle_interval = max(self.size_a, self.size_b)
        self.shuffle_cnt = 0

    def __getitem__(self, index):
#         if not index % self.shuffle_interval:
#         if np.random.random() < 0.001:
#             self.shuffle_cnt += 1
#             np.random.shuffle(self.imlist_a)
#             np.random.shuffle(self.imlist_b)

        impath_a = self.imlist_a[index % self.size_a]
        img_a = self.loader(os.path.join(self.dataroot, impath_a))

        impath_b = self.imlist_b[index % self.size_a]
        img_b = self.loader(os.path.join(self.dataroot, impath_b))

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b, impath_a, impath_b

    def __len__(self):
        return max(self.size_a, self.size_b)


###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class ImageFolder(data.Dataset):
    def __init__(self, root, config, transform=None, return_paths=False, loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.config = config
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.config['split']:
            width = self.config['crop_image_width']
            img_a, img_b = img[..., width:], img[..., :width]
            return img_a, img_b, path

        return img, path

    def __len__(self):
        return len(self.imgs)


class UnpairedImageFolder(data.Dataset):
    def __init__(self, dataroot, phase, transform=None, serial_batches=False, loader=default_loader):
        self.transform = transform
        self.serial_batches = serial_batches
        self.loader = loader

        self.dir_A = os.path.join(dataroot, phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(dataroot, phase + 'B')  # create a path '/path/to/data/trainB'

        self.paths_a = sorted(make_dataset(self.dir_A))
        self.paths_b = sorted(make_dataset(self.dir_B))
        self.size_a = len(self.paths_a)
        self.size_b = len(self.paths_b)

    def __getitem__(self, index):
        path_a = self.paths_a[index % self.size_a]  # make sure index is within then range
        if self.serial_batches:  # make sure index is within then range
            index_B = index % self.size_b
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.size_b - 1)
        path_b = self.paths_b[index_B]

        img_a = self.loader(path_a)
        img_b = self.loader(path_b)

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b, path_a, path_b

    def __len__(self):
        return max(self.size_a, self.size_b)


def get_transform(load_size, crop, height, width, flip):
    transform_list = []
    if load_size:
        transform_list += [transforms.Resize((load_size, load_size), Image.BICUBIC)]
    if crop:
        transform_list += [transforms.RandomCrop((height, width))]
    if flip:
        transform_list += [transforms.RandomHorizontalFlip()]

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)
    return transform


def get_data_loader(config, dataset_cls, shuffle, is_train=True, **kwargs):  # TODO : kind of interface
    load_size = config['load_size']
    crop = config['crop']
    height = config['crop_image_height']
    width = config['crop_image_width']
    flip = config['flip']
    num_workers = config['num_workers']

    batch_size = config['batch_size'] if is_train else config['batch_size_test']

    transform = get_transform(load_size, crop, height, width, flip)

    dataset = dataset_cls(config=config, transform=transform, **kwargs)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers)
    return loader, dataset
