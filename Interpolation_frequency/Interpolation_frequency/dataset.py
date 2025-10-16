import torch
from torch.utils.data import Dataset
import numpy as np

class SineMixDataset(Dataset):
    """
    Toy dataset: mixtures of sine waves at specified / sampled frequencies.

    Each trajectory:
      x(t) = sum_{k=1..K} A_k * sin(2π f_k t + φ_k) + ε(t)

    Returns per sample (univariate D=1 by default):
      - values: (T, D)          noisy dense observations
      - ground_truth: (T, D)    clean dense signal
      - positions: (1, T)       time grid in [0, duration]
      - mask: (T, D)            True where observed, False = missing
      - obs_indices: (n_obs,)   observed time indices (numpy array)

    Notes:
      * mask=True means OBSERVED (aligns with your trainer’s convention).
      * Irregular sampling: choose `n_obs` per sample or a range.
    """
    def __init__(
        self,
        n=2000,
        T=50,
        duration=1.0,
        # frequency control (Hz == cycles per duration unit)
        freqs=None,                 # e.g. [1,3,5,7,9,12,15,18,21]
        n_components=(1, 4),        # random K in [1..4] per sample if tuple, else fixed int
        amp_range=(0.5, 1.0),       # amplitudes A_k ~ U[a,b]
        phase_uniform=True,         # φ_k ~ U[0, 2π]
        noise_std=0.05,             # ε ~ N(0, noise_std^2)
        n_obs=None,                 # if set -> exactly n_obs observed points
        n_obs_range=(3, 10),        # if n_obs is None -> random in [low, high]
        seed=0,
        D=1                          # dimensionality (keep 1 for this experiment)
    ):
        np.random.seed(seed)
        self.seed = seed
        self.n = n
        self.T = T
        self.duration = duration
        self.dt = duration / (T - 1)
        self.fs = 1.0 / self.dt      # sampling rate (samples per duration unit)
        self.nyquist = 0.5 * self.fs # Nyquist frequency

        # default frequency set if none provided (spans low→near-Nyquist-ish)
        if freqs is None:
            # with T=50 over duration=1 → fs=49, Nyquist≈24.5
            self.freqs = np.array([1, 3, 5, 7, 9, 12, 15, 18, 21, 24])
        else:
            self.freqs = np.array(freqs)

        self.n_components = n_components
        self.amp_range = amp_range
        self.phase_uniform = phase_uniform
        self.noise_std = noise_std
        self.n_obs = n_obs
        self.n_obs_range = n_obs_range
        self.D = D

        self.rng = np.random.RandomState(seed)
        self.t = np.linspace(0.0, duration, T)  # positions in [0, duration]

    def __len__(self):
        return self.n

    def _pick_K(self):
        if isinstance(self.n_components, int):
            return self.n_components
        low, high = self.n_components
        return self.rng.randint(low, high + 1)

    def _pick_n_obs(self):
        if self.n_obs is not None:
            return int(self.n_obs)
        low, high = self.n_obs_range
        return self.rng.randint(low, high + 1)

    def __getitem__(self, idx):
        # per-index RNG → deterministic but different per idx
        rng = np.random.RandomState(self.seed + idx)

        # choose K frequencies for this sample
        K = self._pick_K() if not isinstance(self.n_components, int) else self.n_components
        # (Optionally: filter freqs to below Nyquist to avoid aliasing)
        valid_freqs = self.freqs[self.freqs < self.nyquist]
        if len(valid_freqs) < K:
            raise ValueError(f"Not enough valid freqs below Nyquist ({self.nyquist:.2f}).")
        f_idx = rng.choice(len(valid_freqs), size=K, replace=False)
        f_k = valid_freqs[f_idx]

        # amplitudes & phases
        A_k  = rng.uniform(self.amp_range[0], self.amp_range[1], size=K)
        phi_k = rng.uniform(0, 2*np.pi, size=K) if self.phase_uniform else np.zeros(K)

        # clean signal
        x_clean = np.sum([A*np.sin(2*np.pi*f*self.t + p) for A, f, p in zip(A_k, f_k, phi_k)], axis=0)

        # noise
        x_noisy = x_clean + rng.randn(self.T) * self.noise_std

        # irregular observations
        n_obs = self._pick_n_obs() if self.n_obs is None else int(self.n_obs)
        obs_idx = np.sort(rng.choice(self.T, size=n_obs, replace=False))

        # mask = np.zeros((self.T, self.D), dtype=bool)
        # mask[obs_idx, :] = True
        mask              = np.ones(self.T, dtype=bool)
        mask[obs_idx]     = False # Positions masked as true are to be predicted
    
        # pack tensors
        values       = torch.tensor(x_noisy, dtype=torch.float32)[:, None, None, None]          # (T,1)
        ground_truth = torch.tensor(x_clean, dtype=torch.float32)[:, None]       # (T,1)
        mask_t       = torch.tensor(mask, dtype=torch.bool)                         # (T,1)
        positions    = torch.tensor(self.t, dtype=torch.float32).unsqueeze(0)     # (1,T)

        return {
            "values": values,
            "ground_truth": ground_truth,
            "positions": positions,
            "mask": mask_t,
            # "obs_indices": torch.tensor(obs_idx, dtype=torch.long),
            # "freqs_used": torch.tensor(f_k, dtype=torch.float32),
            # "amps_used":  torch.tensor(A_k, dtype=torch.float32),
        }