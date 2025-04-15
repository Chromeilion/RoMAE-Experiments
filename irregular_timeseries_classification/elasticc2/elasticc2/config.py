from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class ElasticcConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='ELASTICC2_',
        env_file='.env',
        extra="ignore"
    )
    dataset_location: str
    model_size: str = Field("RoMA-small")
