import torch
from torchvision import datasets as dset
from torch.utils.data import random_split


class CIFAR10:
    def __init__(self, data_root, train_size=0.9, split_seed=1):
        self.data_root = data_root
        self.train_size = train_size
        self.split_seed = split_seed

    def train(self, transform):
        train_data = dset.CIFAR10(self.data_root, train=True, transform=transform)
        train_size = int(len(train_data) * self.train_size)
        val_size = len(train_data) - train_size
        train_data, _ = random_split(
            train_data,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )
        return train_data

    def val(self, transform):
        train_data = dset.CIFAR10(self.data_root, train=True, transform=transform)
        train_size = int(len(train_data) * self.train_size)
        val_size = len(train_data) - train_size
        _, val_data = random_split(
            train_data,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )
        return val_data

    def test(self, transform):
        test_data = dset.CIFAR10(self.data_root, train=False, transform=transform)
        return test_data


class LSUN:
    def __init__(self, data_root):
        self.data_root = data_root

    def test(self, transform):
        test_data = dset.LSUN(str(self.data_root / "lsun"), "test", transform=transform)
        return test_data


class SVHN:
    def __init__(self, data_root):
        self.data_root = data_root

    def test(self, transform):
        test_data = dset.SVHN(
            str(self.data_root / "svhn"), "test", transform=transform, download=True
        )
        return test_data


DATASETS = {"cifar10": CIFAR10, "lsun": LSUN, "svhn": SVHN}
