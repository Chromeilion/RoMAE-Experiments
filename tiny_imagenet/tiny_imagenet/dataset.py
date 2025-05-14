from pathlib import Path

from torch.utils.data import Dataset
from tinyimagenet import TinyImageNet
import torch
import torchvision.transforms.v2 as v2
from roma.utils import gen_mask


class CustomTinyImagenet(Dataset):
    def __init__(self, inner_ds):
        self.inner_ds = inner_ds
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = v2.Compose([
            v2.Normalize(mean=mean, std=std),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.positions = torch.tensor([(i, j) for i in range(4) for j in range(4)]).T
        self.mask_ratio = 0.75
        self.fake_pad = torch.zeros((1, 16), dtype=torch.bool)

    def __len__(self):
        return len(self.inner_ds)

    def load_sample(self, index):
        value, label = self.inner_ds[index]
        value = self.transform(value)[None, ...]
        return {
            "values": value,
            "positions": self.positions,
            "label": label,
        }

    def __getitem__(self, index):
        sample = self.load_sample(index)
        mask = gen_mask(self.mask_ratio, self.fake_pad, single=True).squeeze()
        sample["mask"] = mask
        return sample
