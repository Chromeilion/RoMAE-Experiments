from roma.model import (RoMAForClassification, RoMAForClassificationConfig,
                        EncoderConfig)
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch.nn as nn

from lsst.dataset import LSSTDataset
from lsst.config import LSSTConfig


def finetune():
    config = LSSTConfig()
    encoder_args = get_encoder_size(config.encoder_size)
    
    model_config = RoMAForClassificationConfig(
        encoder_config=EncoderConfig(**encoder_args),
        tubelet_size=(1, 1, 1),
        dim_output=1,
        n_channels=1,
        n_pos_dims=1
    )
    model = RoMAForClassification.from_pretrained(config.pretrained_checkpoint)
    model.set_loss_fn(nn.MSELoss())
    trainer_config = TrainerConfig(
        warmup_steps=1000,
        epochs=100,
        base_lr=3e-3,
        eval_every=300,
        save_every=300,
        batch_size=16,
        project_name="Example Experiment"
    )
    trainer = Trainer(trainer_config)
    test_dataset = LSSTDataset()
    train_dataset = LSSTDataset()
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
