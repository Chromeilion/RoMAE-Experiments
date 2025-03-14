# RoMA-Experiments
Repository containing experiments for the Rotary Masked Autoencoder.

## Creating an Experiment

To see how to make an experiment, check out the example directory. 
In essence, each experiment is a regular Python package.
The only hardline requirement is that the package needs to be executable, e.g. 
it should have a ```__main__.py``` file which runs the experiment.
With this, it is then possible to utilize the training utilities present in 
RoMA to do multi-node and multi-GPU training.

## Running an Experiment

All experiments may have slightly different requirements in terms of 
environment variables and inputs, this information can be found in each 
experiment's directory.
This section describes the common procedure that all experiments share.

First you should create your Python environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

Next install RoMA. Likely you'll want to install it in editable mode:

```bash
pip install -e roma@git+https://github.com/Chromeilion/RoMA@main#egg=roma
```

Alternatively, if you dont have access to the GitHub repo, you can also just  
install RoMA from a local directory:

```bash
cd path/to/RoMA
pip install -e .
```

Finally, install the experiment as well.
To do this, cd into the directory containing the Python package corresponding 
to that experiment and install it in editable mode:

```bash
pip install -e .
```
Now everything should be set up. 
To run the experiment locally you can simply execute the package.
E.g. for an experiment called "example-experiment":

```bash
python -m example-experiment
```

The CLI might be different for each experiment. Refer to the experiment-specific 
documentation for this.

### Running on Leonardo

When running on Leonardo, before doing the steps above on the login node,
make sure to load the correct modules:

```bash
module load cuda/12.3
module load python/3.11.6--gcc--8.5.0
```

Then you can follow the steps above to get your Python virtualenvironment set up. 

To  submit a job you can utilize the [script](https://github.com/Chromeilion/RoMA/blob/main/scripts/run_experiment.sh) 
provided in the RoMA repo. Before running it, you must create 
a ```.env``` file with all the required variables in the directory you 
are running from. A list of these variables can be found at the top of 
the script. You also have to create a logs directory. Finally, you can run:

```bash
sbatch run_experiment.sh
```

if the ``run_experiment.sh`` file is in a different directory, this isn't a 
problem, you can still run it as follows:

```bash
sbatch path/to/run_experiment.sh
```
