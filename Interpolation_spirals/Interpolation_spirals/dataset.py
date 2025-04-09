####################################################################################################
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import numpy.random as npr
from roma.utils import gen_mask

class SpiralDataset(Dataset):
    """
    Dataset for generating 2D spiral data for interpolation.
    For each spiral, the inputs are:
      - "values": 30 irregularly sampled (x, y) points (from the noisy trajectory)
      - "positions": the 30 corresponding time points (from the first half of the trajectory)
      - "label": the full, densely sampled (x, y) spiral trajectory for the first half (ground truth)
    """
    def __init__(self, nspiral=300, ntotal=150, start=0., stop=6 * np.pi, noise_std=0.1, noise_a=0.02, a=0.0, b=0.3,test=False,extrapolate=False):
        """
        Args:
          nspiral: number of spirals (batch dimension)
          ntotal: total number of datapoints per spiral
          start, stop: range for the spiral's time values (theta)
          noise_std: observation noise standard deviation (for the sampled trajectory)
          noise_a: noise scale for the spiral parameters a and b
          a, b: parameters of the Archimedean spiral (r = a + b * theta)
        """
        self.nspiral = nspiral
        self.ntotal = ntotal
        self.start = start
        self.stop = stop
        self.noise_std = noise_std
        self.noise_a = noise_a
        self.a = a
        self.b = b
        self.extrapolate = extrapolate
        # Generate the full dense trajectories (both original and noisy)
        self.orig_trajs, self.samp_trajs, self.orig_ts = self._generate_spirals()

        # Define the first-half indices (for interpolation, we only consider the first half)
        self.half_index = self.ntotal // 2  # e.g. if ntotal=500, half_index=250
        self.first_half_ts = self.orig_ts[:self.half_index]  # time grid for the first half

        # For each spiral, randomly select 30 time indices (from the first half) as the irregular observations.
        # Using a fixed random state for reproducibility.
        rng = np.random.RandomState(42)
        self.obs_indices = np.stack(
            [np.sort(rng.choice(self.half_index, 30, replace=False)) for _ in range(self.nspiral)],
            axis=0
        )
        self.test = test
        if self.extrapolate:
            self.upto = self.ntotal
        else:
            self.upto = self.half_index

        if self.test:
            if not self.extrapolate:
                self.masks   = [gen_mask(0.6, torch.zeros((1,self.upto),dtype=torch.bool)).squeeze() for i in range(self.nspiral)] 
            else:
                mask_        = torch.zeros((1,self.ntotal),dtype=torch.bool).squeeze()
                mask_[-75:]  = True
                self.masks   = [mask_ for i in range(self.nspiral)]

        # Gen mask: The 0.6 is the fraction of the data masked out. 
    def __len__(self):
        return self.nspiral

    def __getitem__(self, index):
        if self.test:
            mask = self.masks[index]
        else:   
            mask = gen_mask(0.6, torch.zeros((1,75),dtype=torch.bool)).squeeze()
        
        positions = self.orig_ts[:self.upto]  # shape: (30,)
        values =  self.samp_trajs[index, :self.upto, :self.upto]  
        values = torch.tensor(values, dtype=torch.float32)[:, :, None, None].reshape(-1,1,2,1)

        sample = {
            "mask" : mask,
            "values": values,
            "positions": torch.tensor(positions, dtype=torch.float32).reshape(1, -1) ,
        }
        return sample

    def _generate_spirals(self):
        # Set seed for reproducibility of the dense trajectories
        np.random.seed(0)
        # Create a dense time grid for the entire trajectory
        orig_ts = np.linspace(self.start, self.stop, num=self.ntotal)  
        
        # Create slight random variations in spiral parameters a and b per spiral
        aa = npr.randn(self.nspiral) * self.noise_a + self.a  # shape: (nspiral,)
        bb = npr.randn(self.nspiral) * self.noise_a + self.b  # shape: (nspiral,)

        # Generate clockwise spirals
        zs_cw = self.stop + 1. - orig_ts  # shape: (ntotal,)
        rs_cw = aa.reshape(-1, 1) + bb.reshape(-1, 1) * 50. / zs_cw  # shape: (nspiral, ntotal)
        xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
        orig_traj_cw = np.stack((xs, ys), axis=-1)  # shape: (nspiral, ntotal, 2)
        orig_traj_cw = np.flip(orig_traj_cw, axis=1)

        # Generate counter-clockwise spirals
        zs_cc = orig_ts  # shape: (ntotal,)
        rw_cc = aa.reshape(-1, 1) + bb.reshape(-1, 1) * zs_cc  # shape: (nspiral, ntotal)
        xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
        orig_traj_cc = np.stack((xs, ys), axis=-1)  # shape: (nspiral, ntotal, 2)

        # Randomly choose for each spiral whether to use the clockwise or counter-clockwise trajectory
        orig_trajs = []
        np.random.seed(0)  # reset seed for reproducibility of the choice
        for i in range(self.nspiral):
            np.random.seed(i)  # different seed per spiral for consistent selection
            use_cc = bool(npr.rand() > 0.5)
            orig_traj = orig_traj_cc[i] if use_cc else orig_traj_cw[i]
            orig_trajs.append(orig_traj)
        orig_trajs = np.stack(orig_trajs, axis=0)  # shape: (nspiral, ntotal, 2)

        # Create noisy observations from the original trajectory
        samp_trajs = npr.randn(*orig_trajs.shape) * self.noise_std + orig_trajs

        return orig_trajs, samp_trajs, orig_ts

