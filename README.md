# sensorfusion via shallow decoder neural networks
This repository contains work demonstrating the fusion of a low-spatial resolution (but high-temporal resolution) sensor to a high-spatial resolution (but low-spatial resolution) sensor using a shallow decoder neural network map.

## data
Datasets used include a 1-D Kuramoto-Sivashinsky simulation as well as 2-D Shack-Hartmann and Digital Holography sensors for aero-optical metrology, experimental data provided by the Aero Effects Laboratory (AEL) at AFRL. AEL data is not currently provided, but code to generate 1-D Kuramoto-Sivashisnky (KS) equation data is provided under `data/ks/raw/ks.m`. Running this MATLAB file creates a 1.4G file with 98000 snapshots, each with 2048 spatial pixels.

### subsampling into training datasets
To process the KS superset into smaller datasets for experiments, run 
```
python -m src.dataprocessing_ks
```
from the base directory. Use the `--help` flag for subsampling options. A similar script is provided for AEL dataset users.
```
python -m src.dataprocessing_ael [sh_sh/dh_dh/sh_dh]
```

## machine learning
The sensorfusion modules may be run directly from the base directory. Run with the `--help` flag to see information about command line parameters.
```
python -m src.fusion1d
```

### guild (hyperparameter optimization)
Alternatively, use the hyperparameter optimization tool [guild](https://guild.ai) to invoke runs and track results. Here are some example queries.

1) View all operations, run a model with stock settings, look at run details.
```
guild operations
guild run ks:train
guild compare
guild tensorboard
```
2) Run a model with various hidden layer sizes. View results for that model, sorted by loss.
```
guild run ks:train l1='[512,2048]' l2='[32,64,128]'
guild compare -Fo ks:train --min loss
```
3) Run using a variety of learning rates with Bayesian optimization. Suppress video creation to save time. View the last 20 runs.
```
guild run ks:train lr='[0.01:0.1]' --optimizer gp MAKEVIDS=False
guild compare :20
```