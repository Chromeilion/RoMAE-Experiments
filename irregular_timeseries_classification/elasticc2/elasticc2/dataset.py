"""
A simple example dataset.
"""
import torch
from torch.utils.data.dataset import Dataset
import h5py
from roma.utils import gen_mask


class Elasticc2Dataset(Dataset):
    """
    Very simple dataset where the label is just the mean of the input.
    """
    def __init__(self, database_file, split_no: int, split_type: str,
                 mask_ratio: float = 0.5):
        self.file = h5py.File(database_file, 'r')
        self.idxs = self._get_idxs(split_no, split_type)
        self.mask_ratio = mask_ratio

    def _get_idxs(self, split_no: int, split_type: str):
        if split_type == "test":
            return self.file["test"]
        return self.file["_".join([str(split_type), str(split_no)])]

    def __len__(self):
        return self.idxs.shape[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def __getitem__(self, index):
        # A random value between 0 and 100
        idx = self.idxs[index]
        data = torch.tensor(self.file["data"][idx].flatten()[..., None, None, None])
        times = torch.tensor(self.file["time"][idx].flatten())
        pad_mask = (torch.tensor(self.file["mask"][idx]) > 0.5).flatten()
        label = torch.tensor(self.file["labels"][idx])
        bands = torch.arange(0, 6).repeat(data.shape[0]//6)
        positions = torch.stack([bands, times])
        mask = gen_mask(self.mask_ratio, pad_mask[None, ...])

        sample = {
            "values": data,
            "positions": positions,
            "label": label,
            "mask": mask,
        }
        return sample
