# romae/audio/trainer.py
import random
import numpy as np
import torch
import os

from romae.model import RoMAEForPreTraining, RoMAEForPreTrainingConfig, EncoderConfig
from romae.utils import get_encoder_size
from romae.trainer import Trainer, TrainerConfig

from AudioSet.dataset import Audio_InnerDataset, CustomAudioSet
from AudioSet.config import AudioSetConfig


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
