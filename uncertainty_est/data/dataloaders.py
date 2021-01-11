from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets as dset
from torchvision import transforms as tvt
from uncertainty_eval.datasets import get_dataset, DATASETS

from uncertainty_est.data.datasets import ConcatDataset

DATA_ROOT = Path("../data")


def get_dataloader(
    dataset, split, batch_size=32, data_shape=(3, 32, 32), ood_dataset=None, sigma=0.0
):
    train_transform = []
    test_transform = []

    if len(data_shape) == 3:
        img_size = data_shape[-1]
        train_transform.extend(
            [
                tvt.Resize(img_size, Image.BICUBIC),
                tvt.CenterCrop(img_size),
                tvt.Pad(4, padding_mode="reflect"),
                tvt.RandomRotation(15, resample=Image.BICUBIC),
                tvt.RandomHorizontalFlip(),
                tvt.RandomCrop(img_size),
                tvt.ToTensor(),
                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        test_transform.extend(
            [
                tvt.Resize(img_size, Image.BICUBIC),
                tvt.CenterCrop(img_size),
                tvt.ToTensor(),
                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    if sigma > 0.0:
        noise_transform = lambda x: x + sigma * torch.randn_like(x)
        train_transform.append(noise_transform)
        test_transform.append(noise_transform)

    train_transform = tvt.Compose(train_transform)
    test_transform = tvt.Compose(test_transform)

    try:
        ds = DATASETS[dataset](DATA_ROOT)
    except KeyError as e:
        raise ValueError(f'Dataset "{dataset}" not supported') from e

    if split == "train":
        ds = ds.train(train_transform)
    elif split == "val":
        ds = ds.val(test_transform)
    else:
        ds = ds.test(test_transform)

    if ood_dataset is not None:
        try:
            ood_ds_class = DATASETS[ood_dataset]
            if ood_dataset == "gaussian_noise":
                m = 127.5 if len(data_shape) == 3 else 0.0
                s = 60.0 if len(data_shape) == 3 else 1.0
                mean = torch.empty(*data_shape).fill_(m)
                std = torch.empty(*data_shape).fill_(s)
                ood_ds = ood_ds_class(DATA_ROOT, length=len(ds), mean=mean, std=std)
            elif ood_dataset == "uniform_noise":
                l = 0.0 if len(data_shape) == 3 else -5.0
                h = 255.0 if len(data_shape) == 3 else 5.0
                low = torch.empty(*data_shape).fill_(l)
                high = torch.empty(*data_shape).fill_(h)
                ood_ds = ood_ds_class(DATA_ROOT, length=len(ds), low=low, high=high)
            else:
                ood_ds = ood_ds_class(DATA_ROOT)
        except KeyError as e:
            raise ValueError(f'Dataset "{dataset}" not supported') from e

        ood_train = ood_ds.train(train_transform)
        ds = ConcatDataset(ds, ood_train)

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=split == "train",
        drop_last=split == "train",
    )
    return dataloader
