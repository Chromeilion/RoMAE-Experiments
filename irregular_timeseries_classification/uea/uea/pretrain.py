import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import tqdm
from roma.trainer import TrainerConfig, Trainer
from roma.model import RoMAForPreTraining, RoMAForPreTrainingConfig, EncoderConfig, RoMAForClassification
from roma.utils import get_encoder_size
from sklearn.metrics import classification_report

from uea.config import UEAConfig
from uea.dataset import UEADataset


def pretrain():
    """
    Pre-training script which will train RoMAForPreTraining on the data.
    """
    config = UEAConfig()
    datasets = os.listdir(config.dataset_dir)
    for dataset_name in ["LSST"]:#datasets:
        pretrain_on_ds(config.dataset_dir, dataset_name)


def pretrain_on_ds(data_root, dataset_name: str, mask_percentage: float = 0.3):
    top_savedir = Path(f"./all_results")
    top_savedir.mkdir(exist_ok=True)
    savedir = Path(top_savedir/f"{dataset_name}_savedir")
    savedir.mkdir(exist_ok=True)
    train_ds = UEADataset(prefix=data_root, name=dataset_name,
                    mask_ratio=mask_percentage,
                    dataset="train",
                    pretrain_mask_ratio=0.75)
    test_ds = UEADataset(prefix=data_root, name=dataset_name,
                    mask_ratio=mask_percentage,
                    dataset="test",
                    pretrain_mask_ratio=0.75)
    encoder_args = get_encoder_size("RoMA-tiny")
    encoder_config = EncoderConfig(**encoder_args)
    batch_size = 64
    epochs = 800
    trainer_config = TrainerConfig(
        base_lr=3e-4,
        epochs=epochs,
        eval_every=500,
        save_every=500,
        batch_size=batch_size,
        optimizer="adamw",
        project_name=dataset_name,
        optimizer_args={"betas": (0.9, 0.95), "weight_decay": 0.05},
        checkpoint_dir=str(savedir/"pretrain"),
        run_name=f"{dataset_name}_pretrain",
        warmup_steps=int((len(train_ds)*epochs)//batch_size * 0.1),
    )
    model_config = RoMAForPreTrainingConfig(
        encoder_config=encoder_config,
        n_pos_dims=1,
        n_channels=train_ds[0]["values"].shape[1],
        tubelet_size=(1, 1, 1)
    )
    model = RoMAForPreTraining(model_config)
    trainer = Trainer(trainer_config)
    latest_checkpoint = get_latest_checkpoint(savedir/"pretrain")
    trainer.train(
        train_dataset=train_ds,
        test_dataset=test_ds,
        model=model,
        checkpoint=latest_checkpoint
    )
    model = RoMAForClassification.from_pretrained(
        latest_checkpoint,
        dim_output=14
    )
    latest_checkpoint = get_latest_checkpoint(savedir/"finetune")
    epochs = 15
    batch_size=16
    trainer_config = TrainerConfig(
        base_lr=3e-3,
        epochs=epochs,
        gradient_clip=10,
        eval_every=200,
        save_every=200,
        batch_size=batch_size,
        optimizer="sgd",
        project_name=dataset_name,
        optimizer_args={"weight_decay": 0., "momentum": 0.9},
        checkpoint_dir=str(savedir/"finetune"),
        run_name=f"{dataset_name}_pretrain",
        warmup_steps=int((len(train_ds)*epochs)//batch_size * 0.1),
    )
    trainer = Trainer(trainer_config)
    trainer.train(
        train_dataset=train_ds,
        test_dataset=test_ds,
        model=model,
        checkpoint=latest_checkpoint
    )
    all_preds = []
    all_labels = []
    dataloader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=os.cpu_count() - 1,
        pin_memory=True
    )
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch = {key: val.to("cuda") for key, val in batch.items()}
            logit, _ = model(**batch)
            preds = torch.argmax(torch.nn.functional.softmax(logit, dim=1),
                                 dim=1)
            all_labels.extend(list(batch["label"].cpu().numpy()))
            all_preds.extend(list(preds.cpu().numpy()))
    print(classification_report(
        all_labels,
        all_preds,
    ))


def get_latest_checkpoint(savedir):
    latest_checkpoint = None
    if savedir.exists():
        try:
            latest_checkpoint = max(os.listdir(savedir), key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = savedir/latest_checkpoint
        except ValueError:
            latest_checkpoint = None
    return latest_checkpoint