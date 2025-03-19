"""
A simple example dataset.
"""
import torch
from torch.utils.data.dataset import Dataset


class ExampleDataset(Dataset):
    """
    Very simple dataset where the label is just the mean of the input.
    """
    def __init__(self, n_samples: int = 1000):
        self.labels = torch.zeros(n_samples).uniform_(0, 100)

    def __len__(self):
        return self.labels.shape[0]

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
