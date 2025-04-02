import torch
from torch.utils.data.dataset import Dataset


class PositionalDataset(Dataset):
    """
    Dataset that generates a bunch of 1's with ND uniformly distributed
    positions.
    """
    def __init__(self, n_samples: int = 1000, ndim: int = 1,
                 position_range: tuple[float, float] = (0, 1000),
                 seq_len: int = 10):
        self.labels = torch.zeros((n_samples, ndim, seq_len)).uniform_(*position_range)
        self.seq = torch.ones((seq_len, 1, 1, 1))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        lab = self.labels[index]
        return {
            "values": self.seq,
            "positions": lab,
            "label": lab
        }
