from torchvision import datasets as dset
from torchvision import transforms as tvt
from torch.utils.data import DataLoader, random_split


def _wrap_datasets(train, val, test, *args, **kwargs):
    return (
        DataLoader(train, *args, **kwargs),
        DataLoader(val, *args, **kwargs),
        DataLoader(test, *args, **kwargs),
    )

def get_dataloaders(data_root, dataset, transform=None, test_transform=None, batch_size=32):
    default_transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform = default_transform if transform is None else transform
    test_transform = default_transform if test_transform is None else test_transform

    if dataset == "cifar10":
        train_data = dset.CIFAR10(data_root, train=False, transform=test_transform)         
        train_data, val_data = random_split(train_data, lengths=[0.9, 0.1])
        test_data = dset.CIFAR10(data_root, train=False, transform=test_transform)         

    else:
        raise ValueError(f"Dataset \"{dataset}\" not supported")
    
    return _wrap_datasets(train_data, val_data, test_data, batch_size=batch_size)
