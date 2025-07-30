from romae.utils import get_encoder_size
from romae.model import RoMAEForPreTraining, RoMAEForPreTrainingConfig, EncoderConfig
from romae.trainer import Trainer, TrainerConfig

from elasticc2.dataset import Elasticc2Dataset
from elasticc2.config import ElasticcConfig


def pretrain():
    """
    Pre-training script which will train RoMAForPreTraining on the data.
    """
    config = ElasticcConfig()
    encoder_args = get_encoder_size(config.model_size)

    model_config = RoMAEForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=2,
        n_pos_dims=2,
        use_cls=False
    )
    print("Training only on first fold")
    n_folds = 1
    for fold in range(n_folds):
        print(f"Training on fold {fold}")
        model = RoMAEForPreTraining(model_config)
        trainer_config = TrainerConfig(
            warmup_steps=config.pretrain_warmup_steps,
            checkpoint_dir="checkpoints-pretrain-fold-"+str(fold),
            epochs=config.pretrain_epochs,
            base_lr=config.pretrain_lr,
            eval_every=config.pretrain_eval_every,
            save_every=config.pretrain_save_every,
            optimizer_args=config.pretrain_optimargs,
            batch_size=config.pretrain_batch_size,
            project_name=config.project_name,
            gradient_clip=config.pretrain_grad_clip,
            lr_scaling=True
        )
        trainer = Trainer(trainer_config)
        with (
            Elasticc2Dataset(config.dataset_location, split_no=fold,
                             split_type="validation") as test_dataset,
            Elasticc2Dataset(config.dataset_location, split_no=fold,
                             split_type="training",
                             gaussian_noise=config.gaussian_noise) as train_dataset
        ):
            trainer.train(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                model=model,
            )
