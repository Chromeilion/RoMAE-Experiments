from roma.model import (RoMAForPreTraining,RoMAForPreTrainingConfig, EncoderConfig)
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch.nn as nn

from Interpolation_spirals.dataset import SpiralDataset


def pretrain():
    # Let's use the tiny model:
    encoder_args = get_encoder_size("RoMA-tiny")

    model_config = RoMAForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        tubelet_size=(1, 2, 1),
        n_channels=1,
        n_pos_dims=1
    )

    # HAve beelow in the ENV (not everything)
    model          = RoMAForPreTraining(model_config)
    model.set_loss_fn(nn.MSELoss())
    trainer_config = TrainerConfig(project_name="Interpolation_Electricity")

    trainer       = Trainer(trainer_config)
    test_dataset  = SpiralDataset(nspiral=100)
    train_dataset = SpiralDataset(nspiral=200)
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
