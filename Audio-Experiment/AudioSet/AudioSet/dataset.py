# romae/audio/audio_dataset.py
import torch
from torch.utils.data.dataset import Dataset
import torchaudio
import json
import numpy as np
from romae.utils import gen_mask  

class Audio_InnerDataset(Dataset):
    """Reads {"data": [{"wav": "...", "label": <optional>}, ...]}"""
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data_json = json.load(f)
        self.items = data_json["data"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        #extract waveform + sampling rate from the .wav file
        wav, sr = torchaudio.load(d["wav"])
        #assign a fake label to the waveform
        label = d.get("label", -1)
        return wav, sr, label


audio_conf = {
    'num_mel_bins': 128,       # Number of Mel frequency bins
    'target_length': 1024,     # Target number of time frames per audio clip
    'mean': -4.2677,           # Dataset mean for normalization (precomputed)
    'std': 4.5685,             # Dataset std for normalization (precomputed)
    'mixup': 0.0,              # Mixup augmentation rate (set to 0.0 if not needed)
    'noise': False,            # Noise addition (set True if desired)
    'skip_norm': False,        # Whether to skip normalization (usually False)
    'mode': 'train',           # Dataset mode ('train' or 'eval')
    'dataset': 'AudioSet'      # Name of your dataset
}


class CustomAudioSet(Dataset):
    """
    Dataset class for audio spectrograms.
    Returns dict with keys: values, positions, mask, label.
    """
    def __init__(
        self,
        inner_ds: Dataset,
        audio_conf: dict,
        mean_std_path: str | tuple | None = None,
        mask_ratio: float = 0.75,
    ):
        self.inner_ds = inner_ds
        self.audio_conf = audio_conf

        # Audio params
        self.melbins = audio_conf.get("num_mel_bins", 128)
        self.target_length = audio_conf.get("target_length", 1024)
        self.frame_shift = audio_conf.get("frame_shift", 10)
        self.window_type = audio_conf.get("window_type", "hanning")
        self.use_energy = audio_conf.get("use_energy", False)
        self.dither = audio_conf.get("dither", 0.0)

        # Normalization stats: allow .pt or (mean.npy, std.npy)
        mean = audio_conf.get("mean", -4.2677)
        std = audio_conf.get("std", 4.5685)
        if mean_std_path is not None:
            if isinstance(mean_std_path, str) and mean_std_path.endswith(".pt"):
                stats = torch.load(mean_std_path, map_location="cpu")
                mean, std = stats["mean"], stats["std"]         # [mel]
            elif isinstance(mean_std_path, (list, tuple)) and len(mean_std_path) == 2:
                mean, std = np.load(mean_std_path[0]), np.load(mean_std_path[1])
            else:
                raise ValueError("mean_std_path must be .pt or a (mean.npy, std.npy) tuple")
        self.norm_mean = torch.as_tensor(mean, dtype=torch.float32).view(-1)
        self.norm_std = torch.as_tensor(std, dtype=torch.float32).view(-1)
        if self.norm_mean.numel() == 1:
            self.norm_mean = self.norm_mean.repeat(self.melbins)
            self.norm_std = self.norm_std.repeat(self.melbins)

        # Patch grid and positions (match n_pos_dims=2)
        self.freq_patches = self.melbins // 16
        self.time_patches = self.target_length // 16
        assert self.target_length % 16 == 0, "target_length must be divisible by tubelet_t"
        assert self.melbins % 16 == 0, "melbins must be divisible by tubelet_f"
        self.total_patches = self.freq_patches * self.time_patches
        self.positions = torch.tensor( [(i, j) for i in range(self.time_patches) for j in range(self.freq_patches)]).T  # [2, total_patches]
        self.mask_ratio = mask_ratio
        self.fake_pad = torch.zeros((1, self.total_patches), dtype=torch.bool)

    def __len__(self):
        return len(self.inner_ds)

    def _compute_fbank(self, waveform, sr):
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=self.use_energy,
            window_type=self.window_type,
            num_mel_bins=self.melbins,
            dither=self.dither,
            frame_shift=self.frame_shift,
        )  # [T, mel]
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
        elif p < 0:
            fbank = fbank[: self.target_length, :]
        return fbank

    def _normalize(self, fbank):
        # keep parity with your existing code
        return (fbank - self.norm_mean) / (2 * self.norm_std)

    def load_sample(self, index: int):
        waveform, sr, label = self.inner_ds[index]
        if waveform.shape[0] > 1:
           waveform = waveform.mean(dim=0, keepdim=True) 
        waveform = waveform - waveform.mean()

        fbank = self._compute_fbank(waveform, sr)
        fbank = self._normalize(fbank).to(torch.float32)
        values = fbank.unsqueeze(1).unsqueeze(-1)  # [T, 1, mel,1]
        return {
            "values": values,
            "positions": self.positions,
            "label": torch.tensor(label) if not isinstance(label, torch.Tensor) else label,
        }

    def __getitem__(self, index: int):
        sample = self.load_sample(index)
        mask = gen_mask(self.mask_ratio, self.fake_pad, single=True).squeeze()  # [total_patches]
        sample["mask"] = mask
        return sample
