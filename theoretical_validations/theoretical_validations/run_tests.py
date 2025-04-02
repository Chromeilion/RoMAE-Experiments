from roma.model import (RoMAForClassification, RoMAForClassificationConfig,
                        EncoderConfig, InterpolationHead)
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch.nn as nn

from theoretical_validations.dataset import PositionalDataset
from theoretical_validations.utils import PositionReconstructionHead


def run_tests():
    tests = [
#        {
#            "ndim": 1,
#            "position_range": (1, 1000),
#            "seq_len": 20,
#            "use_cls": True,
#            "d_model": 198 * 4
#        },
#        {
#            "ndim": 1,
#            "position_range": (1, 1000),
#            "seq_len": 20,
#            "use_cls": True,
#            "d_model": 198*2
#        },
        {
            "ndim": 1,
            "position_range": (1, 100),
            "seq_len": 20,
            "use_cls": True,
            "d_model": 198,
            "rope_p": 0.5
        }
    ]
    for test in tests:
        run_single_test(**test)


def run_single_test(ndim: int, position_range: tuple[float, float],
                    seq_len: int, use_cls: bool, d_model: int, rope_p: float = 0.75,
                    npoints: int = 10000):
    batch_size = 32
    epochs = 10
    # Let's use the tiny model:
    encoder_args = get_encoder_size("RoMA-tiny")
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
        base_lr=3e-3,
        optimizer="adamw",
        eval_every=int(0.1*(npoints/batch_size)*epochs),
        save_every=int(0.1*(npoints/batch_size)*epochs),
        batch_size=batch_size,
        project_name="Theoretical Validations"
    )
    trainer = Trainer(trainer_config)
    train_dataset = PositionalDataset(
        n_samples=npoints,
        position_range=position_range,
        seq_len=seq_len,
        ndim=ndim
    )
    test_dataset = PositionalDataset(
        n_samples=int(0.2*npoints),
        position_range=position_range,
        seq_len=seq_len,
        ndim=ndim
    )
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
