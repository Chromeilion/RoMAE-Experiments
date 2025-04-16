from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Any


class ElasticcConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='ELASTICC2_',
        env_file='.env',
        extra="ignore"
    )
    dataset_location: str
    model_size: str = Field("RoMA-small")
    pretrain_epochs: int = Field(100)
    pretrain_lr: float = Field(3e-3)
    pretrain_warmup_steps: int = 4000
    pretrain_batch_size: int = Field(128)
    project_name: str = Field("Elasticc2")
    pretrain_optimargs: dict[str, Any] = {"betas": (0.9, 0.95),
                                          "weight_decay": 0.01}
