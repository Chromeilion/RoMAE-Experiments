from romae.model import (RoMAEForClassification, RoMAEForClassificationConfig,
                         EncoderConfig)
from romae.trainer import Trainer, TrainerConfig
from romae.utils import get_encoder_size
import torch.nn as nn

from example_experiment.dataset import ExampleDataset


def finetune():
    # Let's use the tiny model:
    encoder_args = get_encoder_size("RoMAE-tiny")

    model_config = RoMAEForClassificationConfig(
        encoder_config=EncoderConfig(**encoder_args),
        tubelet_size=(2, 1, 1),
        dim_output=1,
        n_channels=1,
        n_pos_dims=1,
    )
    model = RoMAEForClassification(model_config)
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
    test_dataset = ExampleDataset()
    train_dataset = ExampleDataset()
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
