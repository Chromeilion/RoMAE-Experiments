from pydantic_settings import BaseSettings, SettingsConfigDict


class LSSTConfig(BaseSettings):
    """
    RoMA base configuration, shared by RoMAForClassification and
    RoMAForInterpolation.
    """
    model_config = SettingsConfigDict(
        env_prefix='EXPERIMENT_LSST_',
        env_file='.env',
        extra="ignore",
        env_nested_delimiter='__'
    )
    dataset_dir: str
    pretrained_checkpoint: str
    encoder_size: str = "RoMA-tiny"
