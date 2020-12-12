from typing import List

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets: List[Dataset] = datasets

    def __len__(self):
        return min([len(ds) for ds in self.datasets])

    def __getitem__(self, idx):
        return [ds[idx] for ds in self.datasets]
