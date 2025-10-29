# Example Experiment

This is an example experiment utilizing RoMAE. 
To adapt this to your needs, you can copy and paste this directory and change 
the following:

1. Rename the directories example and example/example to the name of your experiment
2. Update the pyproject.toml file, specifically lines 7, 11, 13, and 28
3. Update this file, adding a brief description of how to run the experiment.

This package has 3 subcommands: finetune, preprocess, and pretrain.
If you want to run finetuning for example, you can run:

```bash
python -m example_experiment finetune
```

With the Leonardo script, you should set ```EXPERIMENT_PACKAGE``` to ```example``` in the 
.env file, and then if you wanna run finetuning you can do:

```bash
sbatch run_experiment.sh finetune
```
