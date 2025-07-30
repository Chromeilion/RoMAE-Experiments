"""
A simple example dataset.
"""
import torch
from torch.utils.data.dataset import Dataset


class PerfDataset(Dataset):
    """
    Very simple dataset where the label is just the mean of the input.
    """
    def __init__(self, im_res: tuple[int, int, int] = (1, 3, 224, 224),
                 size: int = 10000,
                 mask_ratio: float = 0.75):
        self.rand_im = torch.zeros(im_res).uniform_()
        tokens_per_dim = 14
        n_dim = 2
        self.size = size
        self.fake_pad = torch.zeros((1, tokens_per_dim**n_dim), dtype=torch.bool)
        self.positions = torch.tensor([(i, j) for i in range(tokens_per_dim) for j in range(tokens_per_dim)]).T
        self.mask = torch.zeros(tokens_per_dim**n_dim, dtype=torch.bool)
        self.mask[:int(mask_ratio * tokens_per_dim**n_dim)] = True

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = {
            "values": self.rand_im,
            "positions": self.positions,
            "mask": self.mask
        }
        return sample
