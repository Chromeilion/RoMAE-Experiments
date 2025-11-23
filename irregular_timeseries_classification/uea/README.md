# UEA Multivariate Time-series Archive Experiment

Trains RoMAE on various datasets from the 
[UEA Multivariate Time-series Archive](https://doi.org/10.48550/arXiv.1811.00075).
Parts of this codebase, especially data loading utilities, are taken from the 
[TSRegression](https://github.com/ChangWeiTan/TSRegression/blob/master/utils) 
repository.

To run the experiment, first install the package:

```bash
pip install .
```

Next, make sure to set the ```EXPERIMENT_UEA_DATASET_DIR``` environment 
variable to the location of the downloaded datasets.
If you dont wish to upload training metrics to wandb make sure to run 
```wandb offline``` in the directory you are running the script from. 
Finally, run the experiment:

```bash
python -m uea train
```

This will generate one directory per experiment and seed containing the 
final results and checkpoints.
