from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

# Allow utilization of tensor cores
torch.backends.cuda.matmul.allow_tf32 = True

class AudioSetConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='AudioSet_',
        env_file='.env',
        extra="ignore"
    )
    dataset_location: str = "/leonardo_work/ICT25_ESP/sdigioia/Audio_data/AudioSet/"
    seed: int = 42
    pretrained_checkpoint: Optional[str] = None
    