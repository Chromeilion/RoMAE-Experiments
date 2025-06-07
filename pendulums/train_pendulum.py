import os
from pathlib import Path
from functools import partial
import random

import tqdm
import numpy as np
from romae.model import RoMAEForClassification, RoMAEForClassificationConfig, EncoderConfig
from romae.trainer import Trainer, TrainerConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pendulum_generation import generate_pendulums


MEAN, STD = 0.002, 0.0021

def main(model_config, trainer_config, filepath = None, sample_rate=0.5, data_random_seed=0,):
    if filepath is None:
        filepath = "./pendulums"
        if not (Path(filepath)/"pend_regression.npz").exists():
            generate_pendulums(filepath, task="regression")

    img_sizes = (24, 24)
    W_p = img_sizes[0] // model_config.tubelet_size[1]
    H_p = img_sizes[1] // model_config.tubelet_size[2]

    train = Pendulum_regression(file_path=filepath,
                                name='pend_regression.npz',
                                mode='train', sample_rate=sample_rate,
                                random_state=data_random_seed,
                                img_patch_w=W_p,
                                img_patch_h=H_p,
                                mean=MEAN,
                                std=STD)
    test = Pendulum_regression(file_path=filepath, name='pend_regression.npz',
                               mode='test', sample_rate=sample_rate,
                               random_state=data_random_seed,
                               img_patch_w=W_p,
                               img_patch_h=H_p,
                               mean=MEAN,
                               std=STD)
    valid = Pendulum_regression(file_path=filepath,
                                name='pend_regression.npz',
                                mode='valid', sample_rate=sample_rate,
                                random_state=data_random_seed,
                                img_patch_w=W_p,
                                img_patch_h=H_p,
                                mean=MEAN,
                                std=STD)

    model = RoMAEForClassification(config=model_config)
    model.set_loss_fn(PendulumMSELoss())
    model.set_head(PendulumHead(emb_dim=model_config.encoder_config.d_model))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Model has {params} parameters")

    trainer = Trainer(trainer_config)
    trainer.set_post_train_hook(partial(evaluate, test=test))
    trainer.train(train, valid, model)


def evaluate(model, run, device, test):
    test_dl = torch.utils.data.DataLoader(
        test,
        batch_size=50,
        shuffle=False,
        num_workers=os.cpu_count()-1
    )
    test_loss = 0.
    for modelargs in test_dl:
        modelargs = {key: value.to(model.cls.device) for key, value in modelargs.items()}
        preds, loss = model(**modelargs)
        test_loss += loss.cpu().item()

    test_loss *= 1e-3
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

class PendulumMSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, logits, targets):
        return self.mse(logits["mean"], targets)


class PendulumGaussianLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian = nn.GaussianNLLLoss()
        self.eps_max = 10
        self.eps_min = 1e-5

    def forward(self, logits, targets):
        pred_mean, pred_variance = logits["mean"], logits["var"]
        return nn.functional.mse_loss(pred_mean, targets)
        mask = None
        assert pred_mean.shape == targets.shape == pred_variance.shape, f'pred_mean {pred_mean.shape} targets {targets.shape} pred_variance {pred_variance.shape}'

        epsilon = 1e-6 * torch.ones_like(pred_mean)
        pred_variance = torch.maximum(pred_variance, epsilon)

        if mask == None:
            mask = torch.ones_like(pred_mean)

        # sum over dimensions
        const = np.log(2 * np.pi)
        sample_dim_time_wise = mask * \
            (torch.log(pred_variance) + torch.square(pred_mean - targets) / pred_variance + const)
        sample_time_wise = 0.5 * torch.sum(sample_dim_time_wise, -1)

        # mean over time steps
        sample_wise = torch.mean(sample_time_wise, 1)

        return torch.mean(sample_wise)


class PendulumHead(nn.Module):
    def __init__(self, emb_dim: int, n_frames: int = 50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_mean = nn.Sequential(
            nn.RMSNorm(emb_dim),
            nn.Linear(emb_dim, 2)
        )
        self.head_var = nn.Sequential(
            nn.RMSNorm(emb_dim),
            nn.Linear(emb_dim, 100)
        )
        self.n_frames = n_frames
        self.sig = nn.Sigmoid()

    def forward(self, logits):
        B = logits.shape[0]
        cls, logits = logits[:, 0], logits[:, 1:]
        frames = logits.reshape(logits.shape[0], self.n_frames, -1, logits.shape[2])
        frame_means = frames.mean(dim=2).squeeze()

        return {"var": self.head_mean(frame_means), "mean": self.head_var(cls).view(B, 50, 2)}


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


class Pendulum_regression(Dataset):
    def __init__(self, file_path, name, mode, img_patch_h: int, img_patch_w: int,
                 mean=None, std=None, sample_rate=0.5, random_state=0):

        data = dict(np.load(os.path.join(file_path, name)))
        train_obs, train_targets, valid_obs, valid_targets, test_obs, test_targets, \
        train_time_points, valid_time_points, test_time_points = subsample(
            data, sample_rate=sample_rate, random_state=random_state)

        self.h = img_patch_h
        self.w = img_patch_w
        self.n_p_per_patch = self.h*self.w
        self.mean, self.std = mean, std

        if mode == 'train':
            self.obs = train_obs
            self.targets = train_targets
            self.time_points = train_time_points
        elif mode == 'valid':
            self.obs = valid_obs
            self.targets = valid_targets
            self.time_points = valid_time_points
        elif mode == 'test':
            self.obs = test_obs
            self.targets = test_targets
            self.time_points = test_time_points
        else:
            raise RuntimeError(f"Mode {mode} not recognised.")

        self.obs = np.ascontiguousarray(
            np.transpose(self.obs, [0, 1, 4, 2, 3]))/255.0

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx, ...].astype(np.float64)).float()
        if self.mean is not None and self.std is not None:
            obs = (obs - self.mean) / self.std
        targets = torch.from_numpy(self.targets[idx, ...].astype(np.float64)).float()
        time_points = torch.from_numpy(self.time_points[idx, ...])

        return {"values": obs, "label": targets, "positions": time_points[None, ...]}


def subsample(data, sample_rate, imagepred=False, random_state=0):
    train_obs, train_targets = data["train_obs"], data["train_targets"]
    valid_obs, valid_targets = data["valid_obs"], data["valid_targets"]
    test_obs, test_targets = data["test_obs"], data["test_targets"]

    seq_length = train_obs.shape[1]
    train_time_points = []
    valid_time_points = []
    test_time_points = []
    n = int(sample_rate*seq_length)

    if imagepred:
        train_obs_valid = data["train_obs_valid"]
        data_components = train_obs, train_targets, train_obs_valid
        train_obs_sub, train_targets_sub, train_obs_valid_sub = [np.zeros_like(x[:, :n, ...]) for x in data_components]

        valid_obs_valid = data["valid_obs_valid"]
        data_components = valid_obs, valid_targets, valid_obs_valid
        valid_obs_sub, valid_targets_sub, valid_obs_valid_sub = [np.zeros_like(x[:, :n, ...]) for x in data_components]

        test_obs_valid = data["test_obs_valid"]
        data_components = test_obs, test_targets, test_obs_valid
        test_obs_sub, test_targets_sub, test_obs_valid_sub = [np.zeros_like(x[:, :n, ...]) for x in data_components]

    else:
        data_components = train_obs, train_targets, valid_obs, valid_targets, test_obs, test_targets
        train_obs_sub, train_targets_sub, valid_obs_sub, valid_targets_sub, test_obs_sub, test_targets_sub = [
            np.zeros_like(x[:, :n, ...]) for x in data_components]

    for i in range(train_obs.shape[0]):
        rng_train = np.random.default_rng(random_state+i+train_obs.shape[0])  # NOTE - maybe we should change seeding to make this more flexible.
        choice = np.sort(rng_train.choice(seq_length, n, replace=False))
        train_time_points.append(choice)
        train_obs_sub[i, ...], train_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [train_obs, train_targets]]
        if imagepred:
            train_obs_valid_sub[i, ...] = train_obs_valid[i, choice, ...]

    for i in range(valid_obs.shape[0]):
        rng_valid = np.random.default_rng(random_state+i+valid_obs.shape[0])  # NOTE - maybe we should change seeding to make this more flexible.
        choice = np.sort(rng_valid.choice(seq_length, n, replace=False))
        valid_time_points.append(choice)
        valid_obs_sub[i, ...], valid_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [valid_obs, valid_targets]]
        if imagepred:
            valid_obs_valid_sub[i, ...] = valid_obs_valid[i, choice, ...]

    for i in range(test_obs.shape[0]):
        rng_test = np.random.default_rng(random_state+i)  # NOTE - maybe we should change seeding to make this more flexible.
        choice = np.sort(rng_test.choice(seq_length, n, replace=False))
        test_time_points.append(choice)
        test_obs_sub[i, ...], test_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [test_obs, test_targets]]
        if imagepred:
            test_obs_valid_sub[i, ...] = test_obs_valid[i, choice, ...]

    train_time_points, valid_time_points, test_time_points = np.stack(
        train_time_points, 0), np.stack(valid_time_points, 0), np.stack(test_time_points, 0)

    if imagepred:
        return train_obs_sub, train_targets_sub, train_time_points, train_obs_valid_sub, \
               valid_obs_sub, valid_targets_sub, valid_time_points, valid_obs_valid_sub, \
               test_obs_sub, test_targets_sub, test_time_points, test_obs_valid_sub
    else:
        return train_obs_sub, train_targets_sub, valid_obs_sub, valid_targets_sub, test_obs_sub, test_targets_sub, \
               train_time_points, valid_time_points, test_time_points


if __name__ == '__main__':
    n_seeds = 20
    for i in tqdm.tqdm(range(n_seeds), desc="Running experiment"):
        seed = 42 + i
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        encoder_config = EncoderConfig(
            d_model=60,
            nhead=2,
            dim_feedforward=30,
            depth=2,
        )
        model_config = RoMAEForClassificationConfig(
            encoder_config=encoder_config,
            tubelet_size=(1, 24, 24),
            pos_encoding="ropend",
            n_pos_dims=1,
            n_channels=1,
            dim_output=2
        )
        trainer_config = TrainerConfig(
            base_lr=3e-4,
            random_seed=seed,
            epochs=50,
            eval_every=500,
            gradient_clip=None,
            save_every=500,
            batch_size=16,
            warmup_steps=1000,
            optimizer="adamw",
            run_name="pendulum_regression_relative",
            project_name="pendulum_regression"
        )
        main(model_config, trainer_config)
