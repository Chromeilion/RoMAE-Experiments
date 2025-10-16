from romae.model import (RoMAEForPreTraining,RoMAEForPreTrainingConfig, EncoderConfig)
from romae.trainer import Trainer, TrainerConfig
from romae.utils import get_encoder_size
import torch.nn as nn

from Interpolation_frequency.dataset import SineMixDataset


def pretrain():
    # Let's use the tiny model:
    encoder_args = get_encoder_size("RoMAE-tiny")
    decoder_args = get_encoder_size("RoMAE-tiny")

    model_config = RoMAEForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config=EncoderConfig(**decoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=1,
        use_cls = True,      # Relative position embedding,
        p_rope_val = 0.75     # Default is 0.75
    )

    # HAve beelow in the ENV (not everything)
    model          = RoMAEForPreTraining(model_config)
    model.set_loss_fn(nn.MSELoss())
    trainer_config = TrainerConfig(project_name="Interpolation_frequency",optimizer="adamw")

    trainer       = Trainer(trainer_config)
    train_dataset  = SineMixDataset(n=1600, T=100, duration=1.0,freqs=[1,5],n_components=2, noise_std=0.1, n_obs=40, n_obs_range=None, seed=42)
    test_dataset   = SineMixDataset(n=400, T=100, duration=1.0,freqs=[1,5],n_components=2, noise_std=0.1, n_obs=40, n_obs_range=None, seed=43)
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
