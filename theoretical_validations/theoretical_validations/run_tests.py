import random
from functools import partial
import os

from roma.model import (RoMAForClassification, RoMAForClassificationConfig,
                        EncoderConfig)
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from theoretical_validations.dataset import PositionalDataset, RelativeDataset
from theoretical_validations.utils import PositionReconstructionHead, RelativeReconstructionHead


def run_tests():
    tiny_args = get_encoder_size("RoMA-tiny")
    small_args = get_encoder_size("RoMA-small")
    base_args = get_encoder_size("RoMA-base")
    large_args = get_encoder_size("RoMA-large")
    model_d_tests = [
        {
            "ndim": 1,
            "position_range": (1, 50),
            "seq_len": 10,
            "use_cls": True,
            "d_model": tiny_args["d_model"],
            "rope_p": 0.75,
            "wandb_project": "Experimental-Validation-d_model=tiny"
        },
        {
            "ndim": 1,
            "position_range": (1, 50),
            "seq_len": 10,
            "use_cls": True,
            "d_model": small_args["d_model"],
            "rope_p": 0.75,
            "wandb_project": "Experimental-Validation-d_model=small"
        },
        {
            "ndim": 1,
            "position_range": (1, 50),
            "seq_len": 10,
            "use_cls": True,
            "d_model": base_args["d_model"],
            "rope_p": 0.75,
            "wandb_project": "Experimental-Validation-d_model=base"
        },
        {
            "ndim": 1,
            "position_range": (1, 50),
            "seq_len": 10,
            "use_cls": True,
            "d_model": large_args["d_model"],
            "rope_p": 0.75,
            "wandb_project": "Experimental-Validation-d_model=large"
        },
        {
            "ndim": 1,
            "position_range": (1, 50),
            "seq_len": 10,
            "use_cls": False,
            "d_model": tiny_args["d_model"],
            "rope_p": 0.75,
            "wandb_project": "Experimental-Validation-d_model=tiny"
        },
        {
            "ndim": 1,
            "position_range": (1, 50),
            "seq_len": 10,
            "use_cls": False,
            "d_model": small_args["d_model"],
            "rope_p": 0.75,
            "wandb_project": "Experimental-Validation-d_model=small"
        },
        {
            "ndim": 1,
            "position_range": (1, 50),
            "seq_len": 10,
            "use_cls": False,
            "d_model": base_args["d_model"],
            "rope_p": 0.75,
            "wandb_project": "Experimental-Validation-d_model=base"
        },
        {
            "ndim": 1,
            "position_range": (1, 50),
            "seq_len": 10,
            "use_cls": False,
            "d_model": large_args["d_model"],
            "rope_p": 0.75,
            "wandb_project": "Experimental-Validation-d_model=large"
        }
    ]
    errors = run_single_test(**{
            "ndim": 1,
            "position_range": (0, 1000),
            "seq_len": 1,
            "use_cls": True,
            "d_model": large_args["d_model"],
            "rope_p": 0.75,
            "run_extra_eval": True,
            "wandb_project": "Experimental-Validation-Relative",
            "optimizer_args": {"momentum": 0.9, "weight_decay": 0.},
            "lr": 5e-7,
            "optimizer": "sgd",
            "int_pos": True
        })
    mean_error = list(torch.stack(errors).mean(dim=0).detach().cpu().numpy())
    std = list(torch.stack(errors).std(dim=0).detach().cpu().numpy())
    print(mean_error)
    print(std)

    for test in model_d_tests:
        run_single_test(**test)


def run_single_test(ndim: int, position_range: tuple[float, float],
                    seq_len: int, use_cls: bool, d_model: int, rope_p: float = 0.75,
                    npoints: int = 20000, wandb_project: str = "Experimental Validation",
                    run_extra_eval: bool = False, epochs: int = 10, optimizer_args: dict = None,
                    optimizer: str = "adamw", lr: float = 5e-4, int_pos: bool = False):
    if optimizer_args is None:
        optimizer_args = {}
    batch_size = 64
    epochs = epochs
    n_runs = 5
    starting_seed = 40
    # Let's use the tiny model:
    encoder_args = get_encoder_size("RoMA-tiny")
    encoder_args["dim_feedforward"] = 4 * encoder_args["d_model"]
    encoder_args["d_model"] = d_model
    model_config = RoMAForClassificationConfig(
        encoder_config=EncoderConfig(**encoder_args),
        tubelet_size=(1, 1, 1),
        n_channels=1,
        n_pos_dims=ndim,
        use_cls=use_cls,
        dim_output=1,
        p_rope_val=rope_p
    )
    train_dataset = PositionalDataset(
        n_samples=npoints,
        position_range=position_range,
        seq_len=seq_len,
        ndim=ndim,
        int_pos=int_pos
    )
    test_dataset = PositionalDataset(
        n_samples=int(0.2 * npoints),
        position_range=position_range,
        seq_len=seq_len,
        ndim=ndim,
        int_pos=int_pos
    )
    errors = []
    for i in range(n_runs):
        torch.manual_seed(starting_seed + i)
        random.seed(starting_seed + i)
        np.random.seed(starting_seed + i)
        model = RoMAForClassification(model_config)
        model.set_loss_fn(nn.MSELoss())
        model.set_head(PositionReconstructionHead(
            d_model=model_config.encoder_config.d_model,
            d_output=ndim,
            layer_norm_eps=model_config.encoder_config.layer_norm_eps,
            head_drop_rate=0.,
            cls=use_cls
        ))
        trainer_config = TrainerConfig(
            warmup_steps=int(0.2*(npoints/batch_size)*epochs),
            epochs=epochs,
            base_lr=lr,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            eval_every=int(0.1*(npoints/batch_size)*epochs),
            save_every=int(0.1*(npoints/batch_size)*epochs),
            batch_size=batch_size,
            project_name=wandb_project,
            gradient_clip=None,
            random_seed=starting_seed+i,
        )
        trainer = Trainer(trainer_config)
        trainer.set_post_train_hook(partial(end_eval_callback, eval_dataset=test_dataset))
        trainer.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
        )
        if run_extra_eval:
            n_test_points = 500
            test_positions = torch.arange(position_range[0]+.5, position_range[1]-.5, (position_range[1]-position_range[0])/n_test_points, device="cuda")[None, None, ...]
            test_vals = torch.ones((1, n_test_points, 1, 1, 1), device="cuda")
            error_p = []
            for i in range(n_test_points):
                test_pos = test_positions[..., i].unsqueeze(-1)
                test_val = test_vals[:, i, ...].unsqueeze(1)
                model_out = model(positions=test_pos, values=test_val)
                error = (test_pos.squeeze() - model_out[0].squeeze()).abs()
                error_p.append(error)
            errors.append(torch.tensor(error_p))
    return errors




def end_eval_callback(model: RoMAForClassification, run, device, eval_dataset):
    model.eval()
    full_loss = 0
    dataloader = DataLoader(eval_dataset, batch_size=1,
                            num_workers=os.cpu_count()-1)
    for batch in tqdm.tqdm(dataloader, desc="Running Final Evaluation"):
        batch = {key: val.to(device) for key, val in batch.items()}
        preds, loss = model(**batch)
        full_loss += loss.detach().cpu().item()
    run.log({"Final MSE": full_loss/len(eval_dataset)})
