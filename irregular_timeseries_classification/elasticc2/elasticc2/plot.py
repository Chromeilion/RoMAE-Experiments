from elasticc2.dataset import Elasticc2Dataset
from elasticc2.config import ElasticcConfig


def plot():
    config = ElasticcConfig()
    with Elasticc2Dataset(
            config.dataset_location, split_no=0,
            split_type="training",
            gaussian_noise=config.gaussian_noise) as ds:
        sample = ds[0]

        ...