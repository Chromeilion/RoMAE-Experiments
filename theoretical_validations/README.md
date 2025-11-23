# Theoretical Validations

This experiment serves to validate some of the theory of the model.
Specifically, it investigates the ability of the model to reconstruct exact 
positional information from the positional embeddings.
This is done by passing in a bunch of 1 values with uniformly distributed 
positions.
We then try to predict the exact positions.

First install this package:

```bash
pip install .
```

Then to run the experiment run:

```bash
python -m theoretical_validations run_tests
```

To make the plot from the paper run:
```bash
python -m theoretical_validations plot
```
