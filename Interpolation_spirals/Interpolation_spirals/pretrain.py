from roma.model import (RoMAForPreTraining,RoMAForPreTrainingConfig, EncoderConfig)
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch.nn as nn
import torch
from Interpolation_spirals.dataset import SpiralDataset
import numpy as np
import random
def pretrain():
    np.random.seed(42)
    torch.random.manual_seed(42)
    random.seed(42)
    # Let's use the tiny model:
    encoder_args = get_encoder_size("RoMA-tiny")
    decoder_args = get_encoder_size("RoMA-tiny")
    
    model_config = RoMAForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config=EncoderConfig(**decoder_args),
        tubelet_size=(1, 2, 1),
        n_channels=1,
        n_pos_dims=1,
        use_cls = False
    )

    # HAve beelow in the ENV (not everything)
    model          = RoMAForPreTraining(model_config)
    model.set_loss_fn(nn.MSELoss())
    trainer_config = TrainerConfig(project_name="Interpolation_spirals",optimizer="adamw")

    trainer       = Trainer(trainer_config)
    test_dataset  = SpiralDataset(nspiral=100,test=True)
    train_dataset = SpiralDataset(nspiral=200)
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )

    