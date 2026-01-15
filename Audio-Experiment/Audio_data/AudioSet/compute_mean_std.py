import torch
from torch.utils.data.dataset import Dataset
import torchaudio
import json
import numpy as np
import sys

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
        wav = d["wav"]
        #assign a fake label to the waveform
        label = d.get("label", -1)
        return wav, label


total_sum = 0.0
total_sq_sum = 0.0
total_frames = 0



train_json="/leonardo_work/ICT25_ESP/sdigioia/Audio_data/AudioSet/bal_train_data.json"
val_json="/leonardo_work/ICT25_ESP/sdigioia/Audio_data/AudioSet/eval_data.json"

json_file=train_json

dataset = Audio_InnerDataset(json_file)

print("I am going to read the wavefiles to compute mean/std value per melbin ")


print(len(dataset), flush=True)


for wavefile, label in dataset:  # however you're loading them
    waveform, sr = torchaudio.load(wavefile)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
    )
    total_sum += fbank.sum(dim=0)
    total_sq_sum += (fbank ** 2).sum(dim=0)
    total_frames += fbank.shape[0]

print("Computation completed")

mean = total_sum / total_frames
std = (total_sq_sum / total_frames - mean**2).sqrt()


torch.save({'mean': mean, 'std': std}, '/leonardo_work/ICT25_ESP/sdigioia/Audio_data/AudioSet/Bal_train_stats.pt')

#torch.save({'mean': mean, 'std': std}, 'Eval_stats.pt')
