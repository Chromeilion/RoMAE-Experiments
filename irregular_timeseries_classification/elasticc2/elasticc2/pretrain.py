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
    decoder_args = get_encoder_size("RoMA-tiny")

    model_config = RoMAForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config=EncoderConfig(**decoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=2
    )
    print("Training only on first fold")
    n_folds = 1
    for fold in range(n_folds):
        print(f"Training on fold {fold}")
        model = RoMAForPreTraining(model_config)
        trainer_config = TrainerConfig(
            warmup_steps=config.pretrain_warmup_steps,
            checkpoint_dir="checkpoints-fold"+str(fold),
            epochs=config.pretrain_epochs,
            base_lr=config.pretrain_lr,
            eval_every=200,
            save_every=200,
            optimizer_args=config.pretrain_optimargs,
            batch_size=config.pretrain_batch_size,
            project_name=config.project_name
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
