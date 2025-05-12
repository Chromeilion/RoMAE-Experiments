import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import v2 
import torch
import numpy as np
from roma.utils import gen_mask
import pandas as pd

class TrainDataset(Dataset):
    def __init__(self, folder_path, transform=None, nmax = None):
        self.folder_path = folder_path
        self.transform = transform
        image_paths = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.startswith("n") and fname.endswith(".JPEG")
        ]
        if nmax is not None: 
            self.nmax = nmax
        else:
            self.nmax = len(image_paths)
        self.image_paths = np.random.choice(image_paths, size = self.nmax, replace = False)
        self.mask_ratio = 0.75
        self.pad_mask = torch.zeros(196, dtype = torch.bool)
        positions = [(x, y) for y in range(1,15) for x in range(1,15)]
        self.positions = torch.tensor(positions).T

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        _pos = image_path[(len(self.folder_path)+2):-5].find('_')
        label = int(image_path[(len(self.folder_path)+2):(len(self.folder_path)+2 + _pos)])
        if self.transform:
            image = self.transform(image)
        mask = gen_mask(mask_ratio=self.mask_ratio, pad_mask = self.pad_mask[None,...], single=True).squeeze()
        output_dict = {'values':image[None,...], 'label':label, 'mask':mask, 'positions':self.positions}
        return output_dict

class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None, nmax = None):
        self.folder_path = folder_path
        self.transform = transform
        self.annotations = pd.read_csv(self.folder_path + 'val_annotations.txt', sep =r'\s+', names = ['img_path','label','c0','c1','c2','c3'])
        if nmax is not None: 
            self.nmax = nmax
        else:
            self.nmax = len(self.annotations['img_path'])
        self.image_paths = list(self.annotations['img_path'])[:nmax] 
        self.labels = list(self.annotations['label'])[:nmax]
        self.mask_ratio = 0.75
        self.pad_mask = torch.zeros(196, dtype = torch.bool)
        positions = [(x, y) for y in range(1,15) for x in range(1,15)]
        self.positions = torch.tensor(positions).T

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(self.folder_path + 'images/' + image_path).convert('RGB')
        _pos = image_path[(len(self.folder_path)+2):-5].find('_')
        label = int(self.labels[idx][1:])
        if self.transform:
            image = self.transform(image)

        mask = gen_mask(mask_ratio=self.mask_ratio, pad_mask = self.pad_mask[None,...], single=True).squeeze()
        output_dict = {'values':image[None,...], 'label':label, 'mask':mask, 'positions':self.positions}
        return output_dict


transform = transforms.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    transforms.ToTensor()
])
