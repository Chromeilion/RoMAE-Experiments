from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

# Allow utilization of tensor cores
torch.backends.cuda.matmul.allow_tf32 = True

class TinyImagenetConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='TINY_IMAGENET_',
        env_file='.env',
        extra="ignore"
    )
    dataset_location: str
    seed: int = 42
    pretrained_checkpoint: Optional[str] = None
    n_classes: int = 20