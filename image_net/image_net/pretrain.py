from roma.model import (RoMAForPreTraining,RoMAForPreTrainingConfig, EncoderConfig)
from roma.trainer import Trainer, TrainerConfig
from roma.utils import get_encoder_size
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from image_net.datasetTiny import TrainDataset, TestDataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True

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
    #%test_dataset  = TestDataset('/leonardo_work/Sis25_trotta/imageNET/data/', transform)
    #%train_dataset = TrainDataset('/leonardo_work/Sis25_trotta/imageNET/data/trainimages/', transform = transform)
    train_dataset = TrainDataset('/home/martinrios/martin/trabajos/imageNet/data/tinyImageNet/tiny-imagenet-200/train/images/', transform = transform)
    test_dataset  = TestDataset('/home/martinrios/martin/trabajos/imageNet/data/tinyImageNet/tiny-imagenet-200/val/', transform)
    
    #%print('Training set')
    #%c = 0
    #%for batch in dataloader:
    #%    x = batch
    #%    c += 1
    #%    if c%100 == 0: print('train', str(c))


    #%print('Testing set')
    #%dataloader = DataLoader(test_dataset, batch_size=128, num_workers = 16)
    #%c = 0
    #%for batch in dataloader:
    #%    x = batch
    #%    c += 1
    #%    if c%100 == 0: print('test', str(c))
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
    )
