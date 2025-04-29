from roma.model import (RoMAForPreTraining,RoMAForPreTrainingConfig, EncoderConfig)
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from image_net.dataset import TrainDataset, TestDataset

transform = transforms.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    transforms.ToTensor()
])

def pretrain():
    # Let's use the tiny model:
    encoder_args = get_encoder_size("RoMA-base")

    model_config = RoMAForPreTrainingConfig(
        encoder_config=EncoderConfig(**encoder_args),
        tubelet_size=(1, 16, 16),
        n_channels=3,
        n_pos_dims=2,
        normalize_targets = True,
        use_cls=False
    )

    # HAve beelow in the ENV (not everything)
    model          = RoMAForPreTraining(model_config)
    model.set_loss_fn(nn.MSELoss())
    trainer_config = TrainerConfig(project_name="Image_net_pretrain")

    trainer       = Trainer(trainer_config)
    test_dataset  = TestDataset('/gpfs/projects/ehpc10/imageNet/', transform, nmax = 1000)
    train_dataset = TrainDataset('/gpfs/projects/ehpc10/imageNet/ILSVRC2012_img_train/images/', transform = transform, nmax = 1000)
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
