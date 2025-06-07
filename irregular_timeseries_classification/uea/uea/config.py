from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class UEAConfig(BaseSettings):
    """
    RoMA base configuration, shared by RoMAForClassification and
    RoMAForInterpolation.
    """
    model_config = SettingsConfigDict(
        env_prefix='EXPERIMENT_UEA_',
        env_file='.env',
        extra="ignore",
        env_nested_delimiter='__'
    )
    dataset_dir: str
    pretrained_checkpoint: Optional[str] = Field(None)
    encoder_size: str = "RoMA-tiny"
