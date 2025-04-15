"""
A simple example dataset.
"""
import torch
from torch.utils.data.dataset import Dataset
from scipy.io import arff


class LSSTDataset(Dataset):
    def __init__(self, dataset_file: str):
        self.dataset = arff.loadarff(dataset_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # A random value between 0 and 100
        mean = self.labels[index]
        # The expected shape is TCHW, so we add in 3 extra dimensions
        value = torch.zeros(100).normal_(mean=mean.item())[:, None, None, None]
        sample = {
            "values": value,
            "positions": torch.zeros((1, 50)),
            "label": mean
        }
        return sample
