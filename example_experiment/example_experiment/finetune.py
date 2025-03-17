from roma.model import RoMAForClassification, RoMAForClassificationConfig
from roma.trainer import Trainer, TrainerConfig
import torch.nn as nn

from example_experiment.example_experiment.dataset import ExampleDataset


def finetune():
    model_config = RoMAForClassificationConfig(
        tubelet_size=(2, 1, 1),
        dim_output=1,
        n_channels=1
    )
    model = RoMAForClassification(model_config)
    model.set_loss_fn(nn.MSELoss())
    trainer_config = TrainerConfig(
        warmup_steps=1000,
        epochs=100,
        base_lr=3e-3,
        eval_every=300,
        save_every=300,
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
