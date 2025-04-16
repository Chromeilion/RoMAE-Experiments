# pylint: disable=E1101
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import model_selection


def union_time(data_loader, classif=False):
    tu = []
    for batch in data_loader:
        if classif:
            batch = batch[0]
        tp = batch[:, :, -1].numpy().flatten()
        for val in tp:
            if val not in tu:
                tu.append(val)
    tu.sort()
    return torch.from_numpy(np.array(tu))
import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticDataset(Dataset):
    """
    Synthetic dataset for generating sparse, irregularly sampled univariate time series.

    Each trajectory is generated as follows:
      - 10 reference time points: r_k = 0.1 * k for k=1,...,10.
      - Reference values: z_k ~ N(0,1) for k=1,...,10.
      - Dense time grid: t_i = 0.02 * i for i=1,...,50.
      - Ground truth (noise-free):
            x_i = (sum_k exp(-alpha*(t_i - r_k)^2) * z_k) / (sum_k exp(-alpha*(t_i - r_k)^2))
      - Observed values:
            x_i (noisy) = x_i + ε, with ε ~ N(0,0.1^2).
      - From the 50 time points, a random subset of between 3 and 10 indices is selected 
        to simulate sparse observations.

    Each sample (trajectory) returns a dictionary with:
        - 'dense_positions': torch.Tensor, shape (50,)
        - 'ground_truth': torch.Tensor, shape (50,) (noise-free signal)
        - 'observed': torch.Tensor, shape (50,) (noisy full trajectory)
        - 'sparse_positions': torch.Tensor, shape (num_obs,)
        - 'sparse_values': torch.Tensor, shape (num_obs,)
        - 'mask': torch.Tensor, shape (50,), bool, True at indices with an observation
    """
    def __init__(self, n=2000, alpha=120.0,num_obs=None, noise_std=0.1, seed=0,test=False):
        np.random.seed(seed)
        self.n = n
        self.alpha = alpha
        self.noise_std = noise_std
        self.num_obs = num_obs
        self.seed = seed
        self.test = test
        self.alpha = alpha
        self.n = n
        self.noise_std = noise_std

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self.num_obs is None:
            # Randomly choose a number of observations between 3 and 10 (inclusive)
            self.num_obs = np.random.randint(3, 11)

        # Set seed for reproducibility
        np.random.seed(self.seed)
        
        # Precompute the dense time grid: t_i = 0.02 * i for i = 1,...,50.
        self.dense_positions = np.array([0.02 * i for i in range(1, 51)])  # shape (50,)
        
        # Define the 10 reference time points: r_k = 0.1 * k for k = 1,...,10.
        self.ref_points = np.array([0.1 * k for k in range(1, 11)])  # shape (10,)
        
        # Pre-compute the fixed weight matrix (shape: 50 x 10)
        # For each dense time point t_i, compute:
        #   w_{ik} = exp(-alpha*(t_i - r_k)^2) / sum_{k'} exp(-alpha*(t_i - r_{k'})^2)
        t = self.dense_positions[:, None]  # shape (50, 1)
        r = self.ref_points[None, :]         # shape (1, 10)
        kernel = np.exp(-self.alpha * (t - r) ** 2)  # shape (50, 10)
        self.W = kernel / np.sum(kernel, axis=1, keepdims=True)  # shape (50, 10)
        
        self.samples = []

        if not self.test:
            for _ in range(self.n):
                # Sample reference values z ~ N(0,1) for each of the 10 reference points
                z = np.random.randn(10)
                
                # Compute the noise-free dense signal (ground truth) using the RBF smoother.
                dense_signal = np.dot(self.W, z)  # shape (50,)
                
                # Add observation noise to generate the noisy full trajectory.
                dense_signal_noisy = dense_signal + np.random.randn(50) * self.noise_std
                
                # Randomly choose a number of observations between 3 and 10 (inclusive)
                

                # Randomly select observation indices (without replacement) from the 50 time points
                obs_indices       = np.sort(np.random.choice(50, size=self.num_obs, replace=False))
                mask              = np.ones(50, dtype=bool)
                mask[obs_indices] = False # Positions masked as true are to be predicted
            

                # Get the sparse positions and values based on the chosen indices
                sparse_positions = self.dense_positions[obs_indices]
                sparse_values = dense_signal_noisy[obs_indices]
                
                sample = {
                    'positions': torch.tensor(self.dense_positions, dtype=torch.float32).reshape(1,50),  # (50,)
                    'ground_truth': torch.tensor(dense_signal, dtype=torch.float32),              # (50,) noise-free
                    'values': torch.tensor(dense_signal_noisy, dtype=torch.float32)[:, None, None, None],              # (50,) noisy full
                    'sparse_positions': torch.tensor(sparse_positions, dtype=torch.float32),      # (num_obs,)
                    'sparse_values': torch.tensor(sparse_values, dtype=torch.float32),            # (num_obs,)
                    'mask': torch.tensor(mask, dtype=torch.bool)                                  # (50,)
                }
                self.samples.append(sample)


        else: # If we are in test mode, over n data we generate n with nobs = 3, 10 and 20 respectively.
                
                for nn in [3,10,20]:
                
                    for _ in range(self.n):
                    
                        # Sample reference values z ~ N(0,1) for each of the 10 reference points
                        z = np.random.randn(10)
                        
                        # Compute the noise-free dense signal (ground truth) using the RBF smoother.
                        dense_signal = np.dot(self.W, z)  # shape (50,)
                        
                        # Add observation noise to generate the noisy full trajectory.
                        dense_signal_noisy = dense_signal + np.random.randn(50) * self.noise_std
                    
                        # Randomly choose a number of observations between 3 and 10 (inclusive)
                        

                        # Randomly select observation indices (without replacement) from the 50 time points
                        obs_indices       = np.sort(np.random.choice(50, size=nn, replace=False))
                        mask              = np.ones(50, dtype=bool)
                        mask[obs_indices] = False # Positions masked as true are to be predicted
                    

                        # Get the sparse positions and values based on the chosen indices
                        sparse_positions = self.dense_positions[obs_indices]
                        sparse_values = dense_signal_noisy[obs_indices]
                        
                        sample = {
                            'positions': torch.tensor(self.dense_positions, dtype=torch.float32).reshape(1,50),  # (50,)
                            'ground_truth': torch.tensor(dense_signal, dtype=torch.float32),              # (50,) noise-free
                            'values': torch.tensor(dense_signal_noisy, dtype=torch.float32)[:, None, None, None],              # (50,) noisy full
                            'sparse_positions': torch.tensor(sparse_positions, dtype=torch.float32),      # (num_obs,)
                            'sparse_values': torch.tensor(sparse_values, dtype=torch.float32),            # (num_obs,)
                            'mask': torch.tensor(mask, dtype=torch.bool)                                  # (50,)
                        }
                        self.samples.append(sample)





        return self.samples[index]
