import random
import os

import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from roma.model import RoMAForClassification
from roma.trainer import Trainer, TrainerConfig
from tinyimagenet import TinyImageNet
from sklearn.metrics import classification_report

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
    train_ds = TinyImageNet(config.dataset_location, split="train")
    val_ds = TinyImageNet(config.dataset_location, split="val")
#    train_ds = [train_ds[i] for i in range(len(train_ds))]
#    val_ds = [val_ds[i] for i in range(len(val_ds))]
    train_ds = CustomTinyImagenet(inner_ds=train_ds, dataaug=True)
    val_ds = CustomTinyImagenet(inner_ds=val_ds, dataaug=False)
    model = RoMAForClassification.from_pretrained(
        checkpoint=config.pretrained_checkpoint,
        dim_output=config.n_classes,
    )
    encoder_config = model.config.encoder_config
    encoder_config.drop_path_rate = 0.1
    model = RoMAForClassification.from_pretrained(
        checkpoint=config.pretrained_checkpoint,
        dim_output=config.n_classes,
        encoder_config=encoder_config,
    )
    model.set_loss_fn(nn.CrossEntropyLoss(label_smoothing=0.1))
    trainer_config = TrainerConfig(
        epochs=25,
        optimizer="adamw",
        optimizer_args={"weight_decay": 0.05, "betas": (0.9, 0.999)},
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
    do_eval(model, val_ds, "cls_rep.txt")


def do_eval(model, test_dataset, output_filename: str):
    model_preds = []
    labels = []
    dl = DataLoader(test_dataset, num_workers=os.cpu_count()//2-1)
    for sample in tqdm.tqdm(dl):
        sample = {key: val.to("cuda") for key, val in sample.items()}
        logits, _ = model(**sample)
        model_preds.append(logits.softmax(dim=-1).argmax(dim=-1).detach().cpu().item())
        labels.append(sample["label"].detach().cpu().item())
    cls_report = classification_report(labels, model_preds, digits=4)
    print(cls_report)
    with open(output_filename, "w") as f:
        f.write(cls_report)
