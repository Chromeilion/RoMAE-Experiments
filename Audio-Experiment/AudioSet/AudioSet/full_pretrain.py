# audio_pretraining.py
import random
import numpy as np
import torch
import os

from romae.model import RoMAEForPreTraining, RoMAEForPreTrainingConfig, EncoderConfig
from romae.utils import get_encoder_size, gen_mask
from romae.trainer import Trainer, TrainerConfig
from torch.utils.data.dataset import Dataset
import torchaudio
import json
 

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


def pretrain(
    config = AudioSetConfig(),
    train_json: str = "/leonardo_work/ICT25_ESP/sdigioia/Audio_data/AudioSet/bal_train_data.json",
    val_json: str = "/leonardo_work/ICT25_ESP/sdigioia/Audio_data/AudioSet/eval_data.json",
    mean_std_path: str = "/leonardo_work/ICT25_ESP/sdigioia/Audio_data/AudioSet/Bal_train_stats.pt",#| tuple | None = None,
    mask_ratio: float = 0.75,
    ):
    
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Build datasets
    audio_conf = {
        "num_mel_bins": 128,
        "target_length": 1024,
        "mean": -4.2677,
        "std": 4.5685,
        "frame_shift": 10,
        "window_type": "hanning",
        "use_energy": False,
        "dither": 0.0,
    }

    train_inner = Audio_InnerDataset(train_json)
    val_inner = Audio_InnerDataset(val_json)

    train_dataset = CustomAudioSet(train_inner, audio_conf, mean_std_path, mask_ratio=mask_ratio)
    test_dataset  = CustomAudioSet(val_inner,   audio_conf, mean_std_path, mask_ratio=mask_ratio)
    encoder_args = get_encoder_size("RoMAE-small")
    encoder_config = EncoderConfig(**encoder_args)
    decoder_config = get_encoder_size("RoMAE-small") #modifica fattaa da me
    decoder_config["depth"] = 2
    # Model config for interpolation-style pretraining
    model_config = RoMAEForPreTrainingConfig(
        encoder_config=encoder_config,   # correct key
        decoder_config=decoder_config,  # default in your model
        tubelet_size=(16, 16, 1),  # [T, F, C] patching
        n_channels=1,
        n_pos_dims=2,              # matches dataset positions shape [2, L]
        normalize_targets=True,
        use_cls=False,
    )

    model = RoMAEForPreTraining(model_config)

    # Trainer config: fill the required fields used by your Trainer
    trainer_config = TrainerConfig(
        project_name="RoMAE-Audio",
        num_dataset_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
        base_lr=1e-4,
        epochs=150,
        batch_size=64,
        eval_every=500,
        save_every=1000,
        optimizer="adamw",
        optimizer_args={"weight_decay": 0.05, "betas": (0.9, 0.95)},
        warmup_steps=1000,
        gradient_clip=1.0,
        lr_schedule="cosine",
        checkpoint_dir="./checkpoints/",
        random_seed=config.seed,
        lr_scaling=True,
    )

    trainer = Trainer(trainer_config)

    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
