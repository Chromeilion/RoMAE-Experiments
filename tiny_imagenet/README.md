# Tiny ImageNet Experiment

Trains RoMAE on the [Tiny ImageNet](http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/yle_project.pdf) 
dataset.
To train the model, set the ```TINY_IMAGENET_DATASET_LOCATION``` env variable 
to the location of the dataset.
Make sure to install the package:

```bash
pip install .
```

Then to do pretraining run:

```bash
python -m tiny_imagenet pretrain
```

For finetuning, set the location of the pre-trained model using 
```TINY_IMAGENET_PRETRAINED_CHECKPOINT``` and run:

```bash
python -m tiny_imagenet finetune
```

To use the exact same hyperparameters as in the paper, source the environment 
files in run_configs. There are a lot of them because we have 5 
pretraining/finetuning runs per ablation, which translates to 30 independent 
runs (and therefore sets of env vars).

More explicitly, to do one run, you can cd into 
```run_configs/romae_no_cls/pretrain/pretrain1```, source the env file,
and run ```python -m tiny_imagenet pretrain```. Then you can go to 
```run_configs/romae_no_cls/finetune/finetune1```, source the env file, 
and run ```python -m tiny_imagenet finetune```.
You might have to change the ```TINY_IMAGENET_PRETRAINED_CHECKPOINT``` env var 
to point to the checkpoint from the pretrain run if your batch size does not 
match ours.