from roma.utils import get_encoder_size
from roma.model import RoMAForPreTraining, RoMAForPreTrainingConfig, EncoderConfig
from roma.trainer import Trainer, TrainerConfig
import torch.nn as nn

from elasticc2.dataset import Elasticc2Dataset
from elasticc2.config import ElasticcConfig


def pretrain():
    """
    Pre-training script which will train RoMAForPreTraining on the data.
    """
    config = ElasticcConfig()
    # Let's use the tiny model:
    encoder_args = get_encoder_size(config.model_size)

    model_config = RoMAForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        tubelet_size=(2, 1, 1),
        dim_output=1,
        n_channels=1
    )
    n_folds = 5
    for fold in range(n_folds):
        print(f"Training on fold {fold}")
        model = RoMAForPreTraining(model_config)
        trainer_config = TrainerConfig(
            warmup_steps=1000,
            checkpoint_dir="checkpoints-fold"+str(fold),
            epochs=100,
            base_lr=3e-3,
            eval_every=300,
            save_every=300,
            batch_size=16,
            project_name="Elasticc2"
        )
        trainer = Trainer(trainer_config)
        with (
            Elasticc2Dataset(config.dataset_location, split_no=fold,
                             split_type="validation") as test_dataset,
            Elasticc2Dataset(config.dataset_location, split_no=fold,
                             split_type="training") as train_dataset
        ):
            trainer.train(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                model=model,
            )
