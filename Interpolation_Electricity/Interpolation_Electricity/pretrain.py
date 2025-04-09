from roma.model import (RoMAForPreTraining,RoMAForPreTrainingConfig, EncoderConfig)
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch.nn as nn
from ucimlrepo import fetch_ucirepo 
from torch.utils.data import random_split
import Interpolation_Electricity.dataset as dataset
import numpy as np

data = np.load("dataset.npy",allow_pickle=True)
# Check for NaNs in the dataset

has_nans = np.any([np.isnan(sample['values']).any() for sample in data])
print(f"Does the data object contain NaNs? {has_nans}")


def pretrain():
    # Let's use the tiny model:
    encoder_args = get_encoder_size("RoMA-small")

    model_config = RoMAForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config=EncoderConfig(**encoder_args),

        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=2
    )

    # Have beelow in the ENV (not everything)
    model          = RoMAForPreTraining(model_config)
    model.set_loss_fn(nn.MSELoss())
    trainer_config = TrainerConfig(project_name="Interpolation_Electricity",optimizer="adamw")
    trainer        = Trainer(trainer_config)
    train_size                  = int(0.8 * len(data))    # 80% for training
    test_size                   = len(data) - train_size  # 20% for testing
    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    colate_fn                   = dataset.CustomCollate()
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        train_collate_fn=colate_fn,
        eval_collate_fn=colate_fn,
    )