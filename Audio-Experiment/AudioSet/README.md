# Audio Experiment

This is an example experiment utilizing RoMA to perform representation learning on Audio data. 

## Getting Started  

### Prepare the Environment
Clone or download this repository and set it as the working directory
Create a virtual environment and install the dependencies.

First you should create your Python environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```
Next install the necessary dependencies
```bash
pip install -r requirements.txt
```

Next install RoMAE. To install it directly from the repo:

##Install RoMAe

```bash
pip install romae@git+https://github.com/Chromeilion/RoMAE@main#egg=romae




This package has 3 subcommands: finetune, preprocess, and pretrain.
If you want to run finetuning on a Slurm cluster for example, you can run:

```bash
sbatch run_experiment.sh finetune
```

## Data Preparation

For pretraining our dataloader requires one file:
* A json file containing path of the audio and corresponding label.
  * Self-supervised pretraining does not  need any label, but our current version of `dataloader.py` needs label information to run, you need to use a dummy label for pretraining data. Below is an example json file.

```json
 {
    "data": [
        {
            "wav": "/data/sls/audioset/data/audio/eval/_/_/--4gqARaEJE_0.000.flac",
            "labels": "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"
        },
        {
            "wav": "/data/sls/audioset/data/audio/eval/_/_/--BfvyPmVMo_20.000.flac",
            "labels": "/m/03l9g"
        },
      // ... many audio files
        {
            "wav": "/data/sls/audioset/data/audio/eval/_/0/-0BIyqJj9ZU_30.000.flac",
            "labels": "/m/07rgt08,/m/07sq110,/t/dd00001"
        }
    ]
}
```

For finetuning our dataloader requires a file more:
* A csv file containing label information. The labels should be consistent with those in the json file.
  * Again, even for self-supervised pretraining, a dummy csv file is needed.
```csv
index,mid,display_name
0,/m/07rwj00,"dog"
1,/m/07rwj01,"rooster"
2,/m/07rwj02,"pig"
...
```

We performed finetuning on different datasets, and we provide in the audio-data the script to download and preprocess each dataset.

The datasets analyzed are:
* Librispeech: reference in 
* AudioSet:  please see [here](https://research.google.com/audioset/download.html).
* ESC-50: described at 

To combine multiple datasets we used the code `src/prep_data/mix_pretraining_data` from the SSAST official repo.

## Self-Supervised Pretraining

