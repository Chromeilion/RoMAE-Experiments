import random
from multiprocessing import Manager

import torch
import numpy as np
from roma.model import RoMAForClassification
from roma.trainer import Trainer, TrainerConfig

from tiny_imagenet.config import TinyImagenetConfig
from tiny_imagenet.dataset import CustomTinyImagenet

def finetune():
    """
    Pre-training script which will train RoMAForPreTraining on the data.
    """

    config = TinyImagenetConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    train_ds = CustomTinyImagenet(
        dataset_location=config.dataset_location,
        split="train",
        cache_dict=shared_dict_train,
    )
    val_ds = CustomTinyImagenet(
        dataset_location=config.dataset_location,
        split="val",
        cache_dict=shared_dict_eval
    )
    model = RoMAForClassification.from_pretrained(
        checkpoint=config.pretrained_checkpoint,
        dim_output=config.n_classes,
    )
    trainer_config = TrainerConfig(
        epochs=15,
        base_lr=1e-3,
        optimizer="adamw",
        optimizer_args={"weight_decay": 0.05, "betas": (0.99, 0.999)},
        project_name="TI Experiment",
        random_seed=config.seed,
        warmup_steps=1000
    )
    trainer = Trainer(trainer_config)
    trainer.train(
        train_dataset=train_ds,
        test_dataset=val_ds,
        model=model,
    )