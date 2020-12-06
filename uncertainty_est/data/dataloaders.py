from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as dset
from torchvision import transforms as tvt

from uncertainty_est.data.datasets import DATASETS

DATA_ROOT = Path("../data")


def get_dataloader(dataset, split, batch_size=32, img_size=32):
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

    try:
        ds = DATASETS[dataset]
    except KeyError as e:
        raise ValueError(f'Dataset "{dataset}" not supported') from e
    ds = ds(DATA_ROOT)

    if split == "train":
        ds = ds.train(train_transform)
    elif split == "val":
        ds = ds.val(test_transform)
    else:
        ds = ds.test(test_transform)

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    return dataloader
