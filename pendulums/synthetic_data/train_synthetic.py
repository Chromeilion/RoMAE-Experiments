import os
from pathlib import Path
from functools import partial

from torch.utils.data import Dataset

from seft.model import SEFT, ModelConfig
from seft.trainer import Trainer, TrainerConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from gen_data import get_synthetic_data


def main(model_config, trainer_config, filepath = None, sample_rate=0.5, data_random_seed=0):
    data = get_synthetic_data((2000, 128))

    train = SyntheticInterp(data=data["train"])
    test = SyntheticInterp(data=data["test"])
    valid = SyntheticInterp(data=data["val"])

    model = SEFT(config=model_config)
#    model.set_loss_fn(PendulumMSELoss())
    model.set_head(InterpolationHead(emb_dim=model_config.d_model))
    trainer = Trainer(trainer_config)
    trainer.set_post_train_hook(partial(evaluate, test=test))
    trainer.train(train, valid, model)


class SyntheticInterp(Dataset):
    def __init__(self, data):
        ...


def evaluate(model, run, test):
    model.eval()
    test_dl = torch.utils.data.DataLoader(
        test,
        batch_size=50,
        shuffle=False,
        num_workers=os.cpu_count()-1
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    test_loss = 0.
    for modelargs in test_dl:
        modelargs = {key: value.to(device) for key, value in modelargs.items()}
        _, loss = model(**modelargs)
        test_loss += loss.cpu().item()

    test_loss /= len(test_dl)
    print(f"Test set loss: {test_loss}")
    run.log({"Test MSE": test_loss})

def calc_mean_std(dataset):
    """Calculate mean and std of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=os.cpu_count()-1
    )
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in dataloader:
        mean += data["values"].mean()
        std += data["values"].std()
        nb_samples += data["values"].shape[0]

    return mean / nb_samples, std / nb_samples


class InterpolationHead(nn.Module):
    def __init__(self, emb_dim: int, n_frames: int = 50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(emb_dim, 2)
        self.n_frames = n_frames

    def forward(self, logits):
        # drop cls token
        logits = logits[:, 1:, :]
        frames = logits.reshape(logits.shape[0], self.n_frames, -1, logits.shape[2])
        frame_means = frames.mean(dim=2).squeeze()
        return self.head(frame_means)


def collate_fn(batch):
    keys_of_interest = ["positions", "values", "label"]
    batch = [{key: i[key] for key in keys_of_interest} for i in batch]
    keys = list(batch[0].keys())
    labels = torch.stack([i["label"] for i in batch])
    keys.remove("label")
    pad_amount = max([i["values"].shape[0] for i in batch])
    p_batch = {key: torch.stack(
        [F.pad(i[key], (0, pad_amount - i[key].shape[-0])) for i in batch]) for key in keys
    }
    p_batch["label"] = labels
    return p_batch


if __name__ == '__main__':
    model_config = ModelConfig(
        d_model=36,
        nhead=2,
        depth=2,
        tubelet_size=(1, 24, 24),
        max_len=10000,
        head_drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        drop_rate=0.,
        pos_encoding="absolute"
    )
    # 2.35
    trainer_config = TrainerConfig(
        base_lr=6e-4,
        epochs=300,
        eval_every=500,
        gradient_clip=5000,
        save_every=500,
        batch_size=32,
        warmup_steps=2000,
        optimizer="adamw",
        run_name="pendulum_regression_absolute",
        project_name="pendulum_regression"
    )
    main(model_config, trainer_config)
    model_config = ModelConfig(
        d_model=36,
        nhead=2,
        depth=2,
        tubelet_size=(1, 24, 24),
        max_len=10000,
        head_drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        drop_rate=0.,
        pos_encoding="relative"
    )
    # 2.35
    trainer_config = TrainerConfig(
        base_lr=6e-4,
        epochs=300,
        eval_every=500,
        gradient_clip=5000,
        save_every=500,
        batch_size=32,
        warmup_steps=2000,
        optimizer="adamw",
        run_name="pendulum_regression_relative",
        project_name="pendulum_regression"
    )
    main(model_config, trainer_config)
