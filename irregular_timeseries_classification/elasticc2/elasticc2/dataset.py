"""
A simple example dataset.
"""
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import h5py
from romae.utils import gen_mask
#import joblib


class Elasticc2Dataset(Dataset):
    """
    Very simple dataset where the label is just the mean of the input.
    """
    # Step size when picking samples from the validation sets
    VAL_RATIO = 64
    FLUX_MEANS = [12.8765,  9.9520, 16.1126, 21.4952, 25.1622, 29.2021]
    FLUX_STDS = [71.4588,  97.3610, 129.6534, 147.7320, 167.2712, 197.8658]
    QT_LOC = "./QT-New_QT-New_md_fold_0.joblib"

    def __init__(self, database_file, split_no: int, split_type: str,
                 mask_ratio: float = 0.5, gaussian_noise: bool = False):
        self.file = h5py.File(database_file, 'r')
        self.noise = gaussian_noise
        self.idxs = self._get_idxs(split_no, split_type)
        self.mask_ratio = mask_ratio
        self.flux_stds = torch.tensor(self.FLUX_STDS)
        self.flux_means = torch.tensor(self.FLUX_MEANS)
#        qt = joblib.load(self.QT_LOC)
#        feat_col = self.file["norm_feat_col"]
#        self.feat_col  = torch.from_numpy(qt.transform(feat_col[:][self.idxs]))

    def get_standardization_vals(self):
        import tqdm
        n_samples = self.file["data"].shape[0]
        means = torch.zeros(6)
        stds = torch.zeros(6)
        for i in tqdm.tqdm(range(n_samples), total=n_samples):
            data = self.file["data"][i]
            mask = self.file["mask"][i]
            for j in range(6):
                means[j] += data[:, j][mask[:, j] > 0.5].mean() / n_samples
                stds[j] += data[:, j][mask[:, j] > 0.5].std() / n_samples

        return means, stds


    def _get_idxs(self, split_no: int, split_type: str):
        match split_type:
            case "test":
                return self.file["test"]
            case "validation":
                return self.file["_".join([str(split_type), str(split_no)])][::self.VAL_RATIO]
            case _:
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
        data = torch.tensor(self.file["data"][idx]).flatten()
        pad_mask = (torch.tensor(self.file["mask"][idx]) > 0.5).flatten()
        alert_mask = (torch.tensor(self.file["mask_alert"][idx]) > 0.5).flatten()
        pad_mask[alert_mask] = True
        times = torch.tensor(self.file["time"][idx].flatten())
        label = torch.tensor(self.file["labels"][idx])
        bands = torch.arange(0, 6).repeat(data.shape[0]//6)
        positions = torch.stack([bands, times])
        data_var = torch.tensor(self.file["data-var"][idx]).flatten()
        data = torch.stack([data, data_var])
        n_nonpad = pad_mask.sum()
        positions = nn.functional.pad(positions[:, pad_mask], (0, positions.shape[1]-n_nonpad)).float()
        data = nn.functional.pad(data[:, pad_mask], (0, data.shape[1]-n_nonpad))[..., None, None].float().swapaxes(0, 1)
        pad_mask[:] = False
        pad_mask[n_nonpad:] = True
        mask = gen_mask(self.mask_ratio, pad_mask[None, ...], single=True).squeeze()
        if self.noise:
            data = data + torch.randn_like(data) * 0.02
        sample = {
            "values": data,
            "positions": positions,
            "label": label,
            "mask": mask,
            "pad_mask": pad_mask
        }
        return sample
