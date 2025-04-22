import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import v2 
import torch

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
        aux_ind = np.random.choice(np.arange(len(self.image_path)), size = self.nmax)
        self.image_paths = image_paths[aux_ind] 
        self.label = 0  # Puedes usar 0 para "LAB", o cambiarlo según tu necesidad

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        _pos = image_path[(len(self.folder_path)+2):-5].find('_')
        self.label = int(image_path[(len(self.folder_path)+2):(len(self.folder_path)+2 + _pos)])
        if self.transform:
            image = self.transform(image)

        return image, self.label

class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None, nmax = None):
        self.folder_path = folder_path
        self.transform = transform
        image_paths = [
            os.path.join(folder_path + '/ILSVRC2012_img_val/', fname)
            for fname in os.listdir(folder_path + '/ILSVRC2012_img_val/')
            if fname.startswith("n") and fname.endswith(".JPEG")
        ]
        if nmax is not None: 
            self.nmax = nmax
        else:
            self.nmax = len(image_paths)
        self.image_paths = image_paths[:nmax] 
        self.labels = np.loadtxt(self.folder_path + '/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt') # Puedes usar 0 para "LAB", o cambiarlo según tu necesidad

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        _pos = image_path[(len(self.folder_path)+2):-5].find('_')
        label = self.labels[idx] 
        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    transforms.ToTensor()
])
