import os

import torch
from romae.model import RoMAEForClassification, RoMAEForClassificationConfig
from romae.utils import load_from_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from elasticc2.dataset import Elasticc2Dataset
from elasticc2.config import ElasticcConfig
from sklearn.metrics import classification_report


def evaluate():
    # Let's use the tiny model:
    config = ElasticcConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on test set")
    model = load_from_checkpoint(config.eval_checkpoint, RoMAEForClassification,
                                 RoMAEForClassificationConfig).to(device).eval()
    all_preds = []
    all_labels = []
    with (
        Elasticc2Dataset(config.dataset_location, split_no=0,
                         split_type="test") as test_dataset,
    ):
        dataloader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            num_workers=os.cpu_count()-1,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {key: val.to(device) for key, val in batch.items()}
                logit, _ = model(**batch)
                preds = torch.argmax(torch.nn.functional.softmax(logit, dim=1), dim=1)
                all_labels.extend(list(batch["label"].cpu().numpy()))
                all_preds.extend(list(preds.cpu().numpy()))
    print(classification_report(
        all_labels,
        all_preds,
        labels=list(range(20)),
        target_names=config.class_names,
        digits=4
    ))
