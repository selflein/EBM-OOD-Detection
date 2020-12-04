from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as dset
from torchvision import transforms as tvt


def _wrap_datasets(train, val, test, *args, **kwargs):
    return (
        DataLoader(train, *args, **kwargs),
        DataLoader(val, *args, **kwargs),
        DataLoader(test, *args, **kwargs),
    )


def get_dataloaders(data_root, dataset, batch_size=32, img_size=32):
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

    if dataset == "cifar10":
        train_data = dset.CIFAR10(data_root, train=False, transform=train_transform)
        train_size = int(len(train_data) * 0.9)
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, lengths=[train_size, val_size])
        val_data.transform = test_transform
        test_data = dset.CIFAR10(data_root, train=False, transform=test_transform)

    else:
        raise ValueError(f'Dataset "{dataset}" not supported')

    return _wrap_datasets(
        train_data,
        val_data,
        test_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
