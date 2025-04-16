from roma.model import (RoMAForPreTraining,RoMAForPreTrainingConfig, EncoderConfig)
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch.nn as nn

from Interpolation_synthetic.dataset import SyntheticDataset


def pretrain():
    # Let's use the tiny model:
    encoder_args = get_encoder_size("RoMA-small")
    decoder_args = get_encoder_size("RoMA-small")

    model_config = RoMAForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config=EncoderConfig(**decoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=1,
        use_cls = False,    # Relative position embedding,
        p_rope_val = 0.75     # Default is 0.75
    )

    # HAve beelow in the ENV (not everything)
    model          = RoMAForPreTraining(model_config)
    model.set_loss_fn(nn.MSELoss())
    trainer_config = TrainerConfig(project_name="Interpolation_synthetic",optimizer="adamw")

    trainer       = Trainer(trainer_config)
    train_dataset  = SyntheticDataset(n=1600, alpha=120.0, num_obs=None,noise_std=0.1, seed=0)
    test_dataset   = SyntheticDataset(n=400, alpha=120.0,  num_obs=None,noise_std=0.1, seed=0)
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
