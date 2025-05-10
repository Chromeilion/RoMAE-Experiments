import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
from roma.trainer import TrainerConfig, Trainer
from roma.model import RoMAForPreTraining, RoMAForPreTrainingConfig, \
    EncoderConfig, RoMAForClassification, RoMAForClassificationConfig
from roma.utils import get_encoder_size
from sklearn.metrics import classification_report
import random
import numpy as np

from uea.config import UEAConfig
from uea.dataset import UEADataset


def pretrain():
    """
    Pre-training script which will train RoMAForPreTraining on the data.
    """
    config = UEAConfig()
    runs = [
        {
            "dataset": "CharacterTrajectories",
            "model": "RoMA-tiny",
            "ft_lr": 8e-3,
            "ft_epochs": 100,
            "ft_batch_size": 16,
            "ft_eval_every": 250,
            "ft_save_every": 250,
            "do_pretrain": True,
            "ft_optimizer": "sgd",
            "ft_gradient_clip": 1,
            "ft_optimizer_args": {"weight_decay": 0., "momentum": 0.9},
            "ft_label_smoothing": 0.1,
            "ft_drop_path": 0.,
            "ft_hidden_drop_rate": 0.,
            "ft_attn_drop_rate": 0.,
            "ft_warmup_percentage": 0.1
        },
        {
            "dataset": "BasicMotions",
            "model": "RoMA-tiny",
            "ft_lr": 1e-2,
            "ft_epochs": 50,
            "ft_batch_size": 8,
            "ft_eval_every": 10,
            "ft_save_every": 10,
            "do_pretrain": True,
            "ft_optimizer": "sgd",
            "ft_gradient_clip": 1,
            "ft_optimizer_args": {"weight_decay": 0., "momentum": 0.9},
            "ft_label_smoothing": 0.,
            "ft_drop_path": 0.,
            "ft_hidden_drop_rate": 0.,
            "ft_attn_drop_rate": 0.,
            "ft_warmup_percentage": 0.1
        },
        {
            "dataset": "LSST",
            "model": "RoMA-tiny",
            "ft_lr": 3e-2,
            "ft_epochs": 15,
            "ft_batch_size": 16,
            "ft_eval_every": 100,
            "ft_save_every": 100,
            "ft_gradient_clip": 10,
            "do_pretrain": True,
            "ft_optimizer": "sgd",
            "ft_optimizer_args": {"weight_decay": 0., "momentum": 0.9},
            "ft_label_smoothing": 0.1,
            "ft_drop_path": 0.2,
            "ft_hidden_drop_rate": 0.2,
            "ft_attn_drop_rate": 0.2,
            "ft_warmup_percentage": 0.1
        },
        {
            "dataset": "Epilepsy",
            "model": "RoMA-tiny",
            "ft_lr": 2e-3,
            "ft_epochs": 150,
            "ft_batch_size": 16,
            "ft_eval_every": 100,
            "ft_save_every": 100,
            "ft_gradient_clip": 1,
            "do_pretrain": True,
            "ft_optimizer": "sgd",
            "ft_optimizer_args": {"weight_decay": 0., "momentum": 0.9},
            "ft_label_smoothing": 0.2,
            "ft_drop_path": 0.2,
            "ft_hidden_drop_rate": 0.2,
            "ft_attn_drop_rate": 0.2,
            "ft_warmup_percentage": 0.1
        },
        {
            "dataset": "Heartbeat",
            "model": "RoMA-tiny",
            "ft_lr": 2e-2,
            "ft_epochs": 30,
            "ft_batch_size": 32,
            "ft_eval_every": 50,
            "ft_save_every": 50,
            "ft_gradient_clip": 2,
            "do_pretrain": True,
            "ft_optimizer": "sgd",
            "ft_optimizer_args": {"weight_decay": 0., "momentum": 0.9},
            "ft_label_smoothing": 0.,
            "ft_drop_path": 0.,
            "ft_hidden_drop_rate": 0.,
            "ft_attn_drop_rate": 0.,
            "ft_warmup_percentage": 0.1
        }
    ]
    for run in runs:
        for i in [27, 42, 1024]:
            dataset_savedir = Path(f"./{run['dataset']}_run_{str(i)}/")
            dataset_savedir.mkdir(exist_ok=True)
            res = pretrain_on_ds(config.dataset_dir, run, dataset_savedir, seed=i+42)
            with open(dataset_savedir/"results.txt", "w") as f:
                f.write(res)
            print(f"Finished {run['dataset']} run {i}")


def pretrain_on_ds(dataset_dir, run, dataset_savedir: Path = Path("."), mask_percentage: float = 0.3, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    train_ds = UEADataset(prefix=dataset_dir, name=run["dataset"],
                          mask_ratio=0.3,
                          dataset="train",
                          pretrain_mask_ratio=0.75)
    test_ds = UEADataset(prefix=dataset_dir, name=run["dataset"],
                         mask_ratio=0.3,
                         dataset="test",
                         pretrain_mask_ratio=0.75)
    top_savedir = dataset_savedir
    top_savedir.mkdir(exist_ok=True)
    savedir = Path(top_savedir/f"{run['dataset']}_savedir")
    savedir.mkdir(exist_ok=True)
    encoder_args = get_encoder_size(run["model"])
    encoder_config = EncoderConfig(**encoder_args)
    if run["do_pretrain"]:
        batch_size = 64
        epochs = 400
        trainer_config = TrainerConfig(
            random_seed=seed,
            base_lr=3e-4,
            epochs=epochs,
            eval_every=500,
            save_every=500,
            batch_size=batch_size,
            optimizer="adamw",
            project_name=run["dataset"],
            optimizer_args={"betas": (0.9, 0.95), "weight_decay": 0.05},
            checkpoint_dir=str(savedir/"pretrain"),
            run_name=f"{run['dataset']}_pretrain",
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
        latest_checkpoint = get_latest_checkpoint(savedir/"pretrain")
        encoder_config.drop_path_rate = run["ft_drop_path"]
        encoder_config.hidden_drop_rate = run["ft_hidden_drop_rate"]
        encoder_config.attn_drop_rate = run["ft_attn_drop_rate"]
        model = RoMAForClassification.from_pretrained(
            latest_checkpoint,
            dim_output=train_ds.inner_ds.class_names.shape[0],
            encoder_config=encoder_config
        )
    else:
        encoder_config.drop_path_rate = run["ft_drop_path"]
        encoder_config.hidden_drop_rate = run["ft_hidden_drop_rate"]
        encoder_config.attn_drop_rate = run["ft_attn_drop_rate"]
        config = RoMAForClassificationConfig(
            encoder_config=encoder_config,
            n_pos_dims=1,
            n_channels=train_ds[0]["values"].shape[1],
            tubelet_size=(1, 1, 1),
            dim_output=train_ds.inner_ds.class_names.shape[0]
        )
        model = RoMAForClassification(config)
    model.set_loss_fn(nn.CrossEntropyLoss(label_smoothing=run["ft_label_smoothing"]))
    latest_checkpoint = get_latest_checkpoint(savedir/"finetune")
    epochs = 20
    batch_size=16
    trainer_config = TrainerConfig(
        random_seed=seed,
        base_lr=run["ft_lr"],
        epochs=run["ft_epochs"],
        gradient_clip=run["ft_gradient_clip"],
        eval_every=run["ft_eval_every"],
        save_every=run["ft_save_every"],
        batch_size=run["ft_batch_size"],
        optimizer=run["ft_optimizer"],
        project_name=run["dataset"],
        optimizer_args=run["ft_optimizer_args"],
        checkpoint_dir=str(savedir/"finetune"),
        run_name=f"{run['dataset']}_finetune",
        warmup_steps=int((len(train_ds)*run["ft_epochs"])//batch_size * run["ft_warmup_percentage"]),
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
    cls_rep = classification_report(all_labels, all_preds, digits=4)
    print(cls_rep)
    return cls_rep


def get_latest_checkpoint(savedir):
    latest_checkpoint = None
    if savedir.exists():
        try:
            latest_checkpoint = max(os.listdir(savedir), key=lambda x: int(x.split("-")[-1]))
            latest_checkpoint = savedir/latest_checkpoint
        except ValueError:
            latest_checkpoint = None
    return latest_checkpoint