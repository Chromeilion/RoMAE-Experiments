# The Pendulum Dataset

This script trains RoMAE on the 
[Pendulum dataset](https://openreview.net/forum?id=Ai8Hw3AXqks).
The dataset generation code is taken from the 
[RKN repository](https://github.com/ALRhub/rkn_share/).
To run the experiment, install the ```pillow``` package and then run the 
```train_pendulum.py``` script:

```bash
python train_pendulum.py
```

This will generate the dataset (which can take some time) and train RoMAE on 
it across 20 seeds. Results are reported individually after training on each 
seed is done and on WandB.