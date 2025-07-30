import time
import json

import torch
from romae.model import (RoMAEForPreTraining, RoMAEForPreTrainingConfig,
                         EncoderConfig)
from romae.trainer import Trainer, TrainerConfig
from romae.utils import get_encoder_size

from performance_experiment.dataset import PerfDataset
from performance_experiment.custom_rope import NDPRope


def run():
    batch_size = 64

    test_dataset = PerfDataset()
    train_dataset = PerfDataset()

    # Let's use the small model:
    encoder_args = get_encoder_size("RoMAE-small")
    decoder_args = get_encoder_size("RoMAE-tiny-shallow")
    model_config_abs = RoMAEForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config=EncoderConfig(**decoder_args),
        tubelet_size=(1, 16, 16),
        n_channels=3,
        n_pos_dims=2,
        pos_encoding="absolute",
        use_cls=False
    )
    model_abs = RoMAEForPreTraining(model_config_abs)

    model_config_rope_reg = RoMAEForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        decoder_config=EncoderConfig(**decoder_args),
        tubelet_size=(1, 16, 16),
        n_channels=3,
        n_pos_dims=2,
        pos_encoding="ropend",
        use_cls=False
    )
    with torch.no_grad():
        sample = torch.stack([test_dataset[0]["positions"] for _ in range(batch_size)])
        model_rope_reg = RoMAEForPreTraining(model_config_rope_reg)
        pos_emb_enc = NDPRope(
            head_dim=model_config_rope_reg.encoder_config.d_model // model_config_rope_reg.encoder_config.nhead,
            positions=sample, B=64, n_dims=2, p=0.75
        )
        pos_emb_dec = NDPRope(
            head_dim=model_config_rope_reg.decoder_config.d_model // model_config_rope_reg.decoder_config.nhead,
            positions=sample, B=64, n_dims=2, p=0.75
        )
        model_rope_reg.encoder_attn_pos_embedding = pos_emb_enc
        model_rope_reg.decoder_attn_pos_embedding = pos_emb_dec
    model_rope_ireg = RoMAEForPreTraining(model_config_rope_reg)

    models = {
        "RoMAE Relative (Regular)": model_rope_reg,
        "RoMAE Absolute":  model_abs,
        "RoMAE Relative (Irregular)": model_rope_ireg,
    }
    trainer_config = TrainerConfig(
        warmup_steps=100,
        epochs=10,
        base_lr=3e-3,
        eval_every=100000,
        save_every=100000,
        batch_size=batch_size,
        project_name="Performance experiment"
    )
    trainer = Trainer(trainer_config)

    results = {}
    for model_type, model in models.items():
        print(f"Running {model_type}...")
        avg_time = run_trial(n_trials=3, model=model, trainer=trainer,
                             train_dataset=train_dataset, test_dataset=test_dataset)
        print(f"Average time for {model_type}: {avg_time} seconds")
        results[model_type] = avg_time

    with open("results.json", "w") as f:
        json.dump(results, f)

def run_trial(n_trials, model, trainer, train_dataset, test_dataset):
    results = []
    for _ in range(n_trials):
        start = time.time()
        trainer.train(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
        )
        results.append(time.time() - start)
        print(f"Trial {len(results)} completed with {results[-1]} seconds.")
    return sum(results) / len(results)
