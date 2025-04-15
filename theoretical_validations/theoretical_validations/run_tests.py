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

from theoretical_validations.dataset import PositionalDataset
from theoretical_validations.utils import PositionReconstructionHead


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
    for test in model_d_tests:
        run_single_test(**test)


def run_single_test(ndim: int, position_range: tuple[float, float],
                    seq_len: int, use_cls: bool, d_model: int, rope_p: float = 0.75,
                    npoints: int = 20000, wandb_project: str = "Experimental Validation"):
    batch_size = 64
    epochs = 10
    n_runs = 5
    starting_seed = 42
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
        ndim=ndim
    )
    test_dataset = PositionalDataset(
        n_samples=int(0.2 * npoints),
        position_range=position_range,
        seq_len=seq_len,
        ndim=ndim
    )
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
            base_lr=5e-4,
            optimizer="adamw",
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
