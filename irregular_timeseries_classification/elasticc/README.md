# ELAsTiCC Classification Experiment

Competing with [this](https://arxiv.org/abs/2405.03078) paper. A bunch of data 
processing code is taken from [this](https://github.com/alercebroker/ATAT) 
repo. ELasTICC was originally announced [here](https://baas.aas.org/pub/2023n2i358p04/release/1).
and [here](https://baas.aas.org/pub/2023n2i117p01/release/1).

## Checkpoints

| Model Size   | Uses Alert <br/> Mask | F-score | Checkpoint                                                                                                    |
|--------------|-----------------------|---------|---------------------------------------------------------------------------------------------------------------|
| tiny-shallow | <center>❌<center/>    | 0.7106  | [elasticc-romae-tiny-shallow.tar.gz](https://pl.cro.moe/romae_checkpoints/elasticc-romae-tiny-shallow.tar.gz) |
| tiny         | <center>❌<center/>    | 0.8029  | [elasticc-romae-tiny.tar.gz](https://pl.cro.moe/romae_checkpoints/elasticc-romae-tiny.tar.gz)                 |
| tiny-shallow | <center>✅<center/>    | 0.6649  | [elasticc-romae-tiny-shallow-noalert.tar.gz](https://pl.cro.moe/checkpoint-8925-tiny-shallow-noalert.tar.gz)  |
| tiny         | <center>✅<center/>    | 0.7205  | [elasticc-romae-tiny-noalert.tar.gz](https://pl.cro.moe/romae_checkpoints/elasticc-romae-tiny-noalert.tar.gz) |

Because of the expenses involved with training RoMAE on the ELAsTiCC dataset,
we provide the fine-tuned model checkpoints. These can be used for 
classification and model evaluation through the ```evaluate``` command.
To provide a closer comparison to ATAT, we also train a version of RoMAE that 
excludes points whose flux is saturated or flagged as likely to be inaccurate 
and provide the checkpoint here. To enable loading the data without flagged 
points, set the ```ELASTICC_APPLY_ALERT_MASK``` environment variable to 
```True```. The same hyperparameters and training procedure have been used 
across all runs as described in the paper.

## Getting Set Up

First install the package:

```bash
pip install .
```

To download and prepare the data run the preprocess command:

```bash
python -m elasticc preprocess
```

The preprocess function is based on code from [ATAT](https://arxiv.org/abs/2405.03078) 
and can take a while to run. It will create a file called elasticc_final.h5, which is
the file we will use later. If you already have this file from running the 
actual ATAT code, you can skip this step.

### Configuration

In the same fashion as the base RoMAE package, we use a Pydantic 
[BaseSettings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
object to store our experiment configuration. This makes it easy to pass 
variables through the environment or a  ```.env``` file.
The environment prefix for this experiment is ```ELASTICC```.
To see all possible variables, take a look at the ```ElasticcConfig``` class 
located in ```elasticc/config.py```. As an example, to set the dataset 
location you can set the ```ELASTICC_DATASET_LOCATION``` environment variable.

## Model Pre-Training/Fine-Tuning

To run pre-training, run the following command:

```bash
python -m elasticc pretrain
```

This will train a RoMAE model from scratch through masked modeling.
Some environment variables such as ```ELASTICC_DATASET_LOCATION``` (which 
points to the ```elasticc_final.h5``` file) must be set for this to work. 
Some training hyperparameters are set through the experiment configuration, 
while others can be set in the way described in the 
[RoMAE](https://github.com/Chromeilion/RoMAE) package README. Afterwards, 
you can use the checkpoints generated during pre-training to run the 
fine-tuning stage:

```bash
python -m elasticc finetune
```

For this make sure to set the ```ELASTICC_PRETRAINED_MODEL``` variable to the 
pre-trained checkpoint folder you wish to use.
Finally, to run evaluation, set the ```ELASTICC_EVAL_CHECKPOINT``` variable 
and run the following command:

```bash
python -m elasticc evaluate
```

## Training on Clusters

For our experiments we ran the training on a compute cluster over 16 GPUs.
You can see how this can be done in the Slurm scripts located in the 
RoMAE repository [here](https://github.com/Chromeilion/RoMAE/blob/main/scripts/run_experiment.sh).
The need for distributed training comes primarily from the large size of the 
dataset and not the size of the model itself; therefore, training would also 
work with less resources, although it will take longer.
