from roma.model import RoMAForClassification, RoMAForClassificationConfig
from roma.trainer import Trainer, TrainerConfig
import torch

from elasticc2.dataset import Elasticc2Dataset
from elasticc2.config import ElasticcConfig


def finetune():
    # Let's use the tiny model:
    config = ElasticcConfig()

    print("Training only on first fold")
    n_folds = 1
    for fold in range(n_folds):
        print(f"Training on fold {fold}")
        model = RoMAForClassification.from_pretrained(
            config.pretrained_model,
            dim_output=config.n_classes
        )
        model.set_loss_fn(
            torch.nn.CrossEntropyLoss(
                weight=torch.tensor(config.class_weights if config.finetune_use_class_weights else None),
                label_smoothing=config.finetune_label_smoothing
            )
        )
        trainer_config = TrainerConfig(
            warmup_steps=config.pretrain_warmup_steps,
            checkpoint_dir="checkpoints-finetune-fold-"+str(fold),
            epochs=config.finetune_epochs,
            base_lr=config.finetune_lr,
            eval_every=config.finetune_eval_every,
            save_every=config.finetune_save_every,
            optimizer_args=config.finetune_optimargs,
            batch_size=config.finetune_batch_size,
            project_name=config.project_name,
            gradient_clip=config.finetune_grad_clip,
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
