import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

class PhysioNetDataset(Dataset):
    """
    PyTorch Dataset for PhysioNet data loaded from a compressed npz file.
    
    The npz file is expected to contain keys 'train' and 'test'. For the validation
    split, we randomly split 20% of the training data.
    
    Each sample is assumed to be a 2D NumPy array of shape (T, 83), where T is the number 
    of time steps. The last dimension is organized as follows:
    
        - Columns 0 to 40: observed feature values (41 features)
        - Columns 41 to 81: corresponding observation mask (1 for observed, 0 for missing)
        - Column 82: time positions (normalized to [0, 1])
    
    The returned sample is a dictionary with:
        - "values": the observed feature values (T, 41)
        - "mask": the observation mask (T, 41)
        - "positions": the time positions as a (1, T) tensor
        - "sparse_mask": a mask randomly selecting 50% of the observed points
    """
    def __init__(self, split="train", data_path="../data/physionet_compressed.npz"):
        data = np.load(data_path)
        if split == "train":
            self.data = data['train']
        elif split == "test":
            self.data = data['test']
        elif split == "val":
            # Create a validation split (20% held out) from training data.
            _, val_data = train_test_split(data['train'], train_size=0.8, random_state=11, shuffle=True)
            self.data = val_data
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load the sample; it is expected to have shape (T, 83)
        sample = self.data[idx]

        # Slice the components.
        values        = torch.tensor(sample[:, :41], dtype=torch.float32)              # (T, 41)
        mask          = torch.tensor(sample[:, 41:82], dtype=torch.bool)               # (T, 41)
        positions     = torch.tensor(sample[:, 82], dtype=torch.float32).unsqueeze(0)  # (1, T)

        # Create a sparse mask that randomly selects 50% of the observed points.
        sparse_mask   = gen_mask(0.5, mask).squeeze() 

        sample_dict = {
            "pad_mask" : mask,               
            "mask"     : sparse_mask,               
            "values"   : values,           # observed feature values (T, 41)
            "positions": positions,     # time positions (1, T)
        }
        return sample_dict
