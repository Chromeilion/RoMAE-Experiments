# finetune_esc50.py

import os, csv, torch, torch.nn as nn
from torch.utils.data.dataset import Dataset

from romae.model import RoMAEForClassification, RoMAEForClassificationConfig, EncoderConfig

from romae.trainer import Trainer, TrainerConfig
from romae.utils import get_encoder_size

from AudioSet.dataset import Audio_InnerDataset, CustomAudioSet
from AudioSet.config import AudioSetConfig



class ESC50_InnerDataset(Dataset):
    """Reads ESC-50 meta/esc50.csv and returns (wav, sr, label) like your Audio_InnerDataset."""
    def __init__(self, esc_root: str, split: str = "train", heldout_fold: int = 5):
        assert split in {"train", "val"}
        meta_csv = os.path.join(esc_root, "meta", "esc50.csv")
        rows = []
        with open(meta_csv, "r") as f:
            r = csv.DictReader(f)
            for x in r:
                x["fold"] = int(x["fold"])
                rows.append(x)
        self.items = [x for x in rows if (x["fold"] != heldout_fold) ] if split == "train" \
                     else [x for x in rows if (x["fold"] == heldout_fold)]
        self.audio_dir = os.path.join(esc_root, "audio")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        import torchaudio
        row = self.items[idx]
        wav_path = os.path.join(self.audio_dir, row["filename"])
        wav, sr = torchaudio.load(wav_path)           # [C, T]
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)           # mono
        label = int(row["target"])                    # 0..49 in ESC-50 CSV
        return wav, sr, label

def finetune_esc50(
    esc_root: str,
    heldout_fold: int = 5,
    mean_std_path: str | tuple | None = None,  # reuse your precomputed stats if you have them
    ):
    # Reuse your audio_conf and shapes from pretraining
    ac = dict(audio_conf)
    ac.update({
        "num_mel_bins": 128,
        "target_length": 1024,
        # If you have a .pt with per-mel mean/std from your pretrain corpus, pass it below
        "mean": ac.get("mean", -4.2677),
        "std":  ac.get("std",  4.5685),
    })

    train_inner = ESC50_InnerDataset(esc_root, "train", heldout_fold)
    val_inner   = ESC50_InnerDataset(esc_root, "val",   heldout_fold)

    # Wrap with your spectrogram + positions builder; turn masking off for classification
    train_ds = CustomAudioSet(train_inner, ac, mean_std_path=mean_std_path, mask_ratio=0.0)
    val_ds   = CustomAudioSet(val_inner,   ac, mean_std_path=mean_std_path, mask_ratio=0.0)

    # Model config must match dataset positions (2D grid) and tubelets from pretraining
    encoder_args = get_encoder_size("RoMAE-small")
    model_cfg = RoMAEForClassificationConfig(
        encoder_config=EncoderConfig(**encoder_args),
        tubelet_size=(16, 16, 1),
        dim_output=50,           # 50 ESC-50 classes
        n_channels=1,
        n_pos_dims=2,            # positions shape is [2, L] in your dataset
    )
    model = RoMAEForClassification(model_cfg)
    model.set_loss_fn(nn.CrossEntropyLoss())

    # Reasonable fine-tuning defaults; adjust to your GPU and Trainer behavior
    tcfg = TrainerConfig(
        project_name="RoMAE-ESC50",
        base_lr=5e-4,
        epochs=60,
        batch_size=32,
        eval_every=200,
        save_every=200,
        warmup_steps=500,
        optimizer="adamw",
        optimizer_args={"weight_decay": 0.05, "betas": (0.9, 0.95)},
        lr_schedule="cosine",
        gradient_clip=1.0,
        checkpoint_dir="./checkpoints_esc50/",
        random_seed=42,
        lr_scaling=True,
    )
    trainer = Trainer(tcfg)

    # Optional: metrics callback if your Trainer supports it
    def metrics_fn(logits, targets):
        preds = logits.argmax(dim=1)
        return {"acc": (preds == targets).float().mean().item()}

    trainer.train(
        train_dataset=train_ds,
        test_dataset=val_ds,
        model=model,
        metrics_fn=metrics_fn if "metrics_fn" in trainer.train.__code__.co_varnames else None,
    )
