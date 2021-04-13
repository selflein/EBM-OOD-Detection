from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as tvt
from uncertainty_eval.datasets import get_dataset as ue_get_dataset

from uncertainty_est.data.datasets import ConcatDataset, ConcatIterableDataset

DATA_ROOT = Path("../data")


def get_dataset(dataset, data_shape=None, length=10_000):
    try:
        ds_class = ue_get_dataset(dataset)

        if data_shape is None:
            data_shape = ds_class.data_shape

        if dataset == "gaussian_noise":
            m = 127.5 if len(data_shape) == 3 else 0.0
            s = 60.0 if len(data_shape) == 3 else 1.0
            mean = torch.empty(*data_shape).fill_(m)
            std = torch.empty(*data_shape).fill_(s)
            ds = ds_class(DATA_ROOT, length=length, mean=mean, std=std)
        elif dataset == "uniform_noise":
            l = 0.0 if len(data_shape) == 3 else -5.0
            h = 255.0 if len(data_shape) == 3 else 5.0
            low = torch.empty(*data_shape).fill_(l)
            high = torch.empty(*data_shape).fill_(h)
            ds = ds_class(DATA_ROOT, length=length, low=low, high=high)
        else:
            ds = ds_class(DATA_ROOT)
    except KeyError as e:
        raise ValueError(f'Dataset "{dataset}" not supported') from e
    return ds, data_shape


def get_dataloader(
    dataset,
    split,
    batch_size=32,
    data_shape=None,
    ood_dataset=None,
    sigma=0.0,
    num_workers=0,
    drop_last=None,
    shuffle=None,
    mutation_rate=0.0,
):
    train_transform = []
    test_transform = []

    unscaled = False
    try:
        ds, data_shape = get_dataset(dataset, data_shape)
    except ValueError as e:
        if "_unscaled" in dataset:
            dataset = dataset.replace("_unscaled", "")
            unscaled = True
            ds, data_shape = get_dataset(dataset, data_shape)
        else:
            raise e

    if len(data_shape) == 3:
        img_size = data_shape[1]
        train_transform.extend(
            [
                tvt.Resize(img_size, Image.BICUBIC),
                tvt.CenterCrop(img_size),
                tvt.Pad(4, padding_mode="reflect"),
                tvt.RandomRotation(15, resample=Image.BICUBIC),
                tvt.RandomHorizontalFlip(),
                tvt.RandomCrop(img_size),
            ]
        )

        test_transform.extend(
            [
                tvt.Resize(img_size, Image.BICUBIC),
                tvt.CenterCrop(img_size),
            ]
        )

        if unscaled:
            unscaled_transform = tvt.Lambda(
                lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).float()
            )
            test_transform.append(unscaled_transform)
            train_transform.append(unscaled_transform)
        else:
            scale_transform = [
                tvt.ToTensor(),
                tvt.Normalize((0.5,) * data_shape[2], (0.5,) * data_shape[2]),
            ]
            test_transform.extend(scale_transform)
            train_transform.extend(scale_transform)

    if sigma > 0.0:
        noise_transform = lambda x: x + sigma * torch.randn_like(x)
        train_transform.append(noise_transform)
        test_transform.append(noise_transform)

    if mutation_rate > 0.0:
        if len(data_shape) == 3:
            mutation_data_shape = (data_shape[2], data_shape[0], data_shape[1])
        else:
            mutation_data_shape = data_shape

        def mutation_transform(x):
            mask = torch.bernoulli(
                torch.empty(mutation_data_shape).fill_(mutation_rate)
            )
            replace = torch.empty(mutation_data_shape).uniform_(-1, 1) * mask
            return x * mask + replace

        train_transform.append(mutation_transform)

    train_transform = tvt.Compose(train_transform)
    test_transform = tvt.Compose(test_transform)

    if split == "train":
        ds = ds.train(train_transform)
    elif split == "val":
        ds = ds.val(test_transform)
    else:
        ds = ds.test(test_transform)

    setattr(ds, "data_shape", data_shape)

    if isinstance(ds, torch.utils.data.IterableDataset):
        shuffle = False
    else:
        shuffle = split == "train" if shuffle is None else shuffle

    if ood_dataset is not None:
        if isinstance(ds, torch.utils.data.IterableDataset):
            ood_ds, _ = get_dataset(ood_dataset, data_shape)
            ds = ConcatIterableDataset(ds, ood_ds.train(train_transform))
        else:
            ood_ds, _ = get_dataset(ood_dataset, data_shape, length=len(ds))

            ood_train = ood_ds.train(train_transform)
            ds = ConcatDataset(ds, ood_train)

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=split == "train" if drop_last is None else drop_last,
    )
    return dataloader
