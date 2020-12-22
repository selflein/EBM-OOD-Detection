from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as dset
from torchvision import transforms as tvt
from uncertainty_eval.datasets.image.datasets import DATASETS

from uncertainty_est.data.datasets import ConcatDataset

DATA_ROOT = Path("../data")


def get_dataloader(
    dataset, split, batch_size=32, img_size=32, ood_dataset=None, sigma=0.0
):
    train_transform = tvt.Compose(
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

    test_transform = tvt.Compose(
        [
            tvt.Resize(img_size, Image.BICUBIC),
            tvt.CenterCrop(img_size),
            tvt.ToTensor(),
            tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if sigma > 0.0:
        noise_transform = tvt.transforms.Lambda(
            lambda x: x + sigma * torch.randn_like(x)
        )
        train_transform.transforms.append(noise_transform)
        test_transform.transforms.append(noise_transform)

    try:
        ds = DATASETS[dataset](DATA_ROOT)
    except KeyError as e:
        raise ValueError(f'Dataset "{dataset}" not supported') from e

    if split == "train":
        ds = ds.train(train_transform)
        if ood_dataset is not None:
            try:
                ood_ds = DATASETS[dataset](DATA_ROOT)
            except KeyError as e:
                raise ValueError(f'Dataset "{dataset}" not supported') from e

            ood_train = ood_ds.train(train_transform)
            ds = ConcatDataset(ds, ood_train)
    elif split == "val":
        ds = ds.val(test_transform)
    else:
        ds = ds.test(test_transform)

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=split == "train",
        drop_last=split == "train",
    )
    return dataloader
