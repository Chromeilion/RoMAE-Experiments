import random
from multiprocessing import Manager

from einops import rearrange
import torch
import numpy as np
from romae.model import RoMAEForPreTraining, RoMAEForPreTrainingConfig, EncoderConfig
from romae.utils import get_encoder_size
from romae.trainer import Trainer, TrainerConfig
from tinyimagenet import TinyImageNet

from tiny_imagenet.config import TinyImagenetConfig
from tiny_imagenet.dataset import CustomTinyImagenet

def pretrain():
    """
    Pre-training script which will train RoMAForPreTraining on the data.
    """
    config = TinyImagenetConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    train_ds = TinyImageNet(config.dataset_location, split="train")
    val_ds = TinyImageNet(config.dataset_location, split="val")
    train_ds = [train_ds[i] for i in range(len(train_ds))]
    val_ds = [val_ds[i] for i in range(len(val_ds))]
    train_ds = CustomTinyImagenet(inner_ds=train_ds)
    val_ds = CustomTinyImagenet(inner_ds=val_ds)
    encoder_args = get_encoder_size("RoMAE-small")
    encoder_config = EncoderConfig(**encoder_args)
    decoder_config = get_encoder_size("RoMAE-small")
    decoder_config["depth"] = 2
    model_config = RoMAEForPreTrainingConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        tubelet_size=(1, 16, 16),
        n_channels=3,
        n_pos_dims=2,
        normalize_targets=True
    )
    model = RoMAEForPreTraining(model_config)
    trainer_config = TrainerConfig(
        epochs=200,
        optimizer="adamw",
        optimizer_args={"weight_decay": 0.05, "betas": (0.9, 0.95)},
        project_name="TI Experiment",
        random_seed=config.seed,
        lr_scaling=False
    )
    trainer = Trainer(trainer_config)
    trainer.train(
        train_dataset=train_ds,
        test_dataset=val_ds,
        model=model,
    )
