# sensorfusion via shallow decoder neural networks

This repository contains work demonstrating the fusion of a low-spatial resolution (but high-temporal resolution) sensor to a high-spatial resolution (but low-spatial resolution) sensor using a shallow decoder neural network map.

Datasets used include a 1-D Kuramoto-Sivashinsky simulation as well as 2-D Shack-Hartmann and Digital Holography sensors for aero-optical metrology, experimental data provided by the Aero Effects Laboratory (AEL) at AFRL. AEL data is not currently provided, but code to generate a KS simulation dataset is contained in this repository.

## datasets

Code to generate data using the 1-D Kuramoto-Sivashisnky (KS) equation is provided under `data/ks/raw/ks.m`. Running this MATLAB file creates a 1.4G simulation data file with 2048 spatial pixels and 98000 snapshots depicting KS flow. AEL data is not currently provided, but see the data directory and read AEL.md for links to references.

To process the KS superset into smaller datasets for experimeriments run 
```
python -m src.dataprocessing_ks
```
from the base directory to subsample the KS data as desired. Run with the `--help` flag for options.

As a note for AEL data users, a similar script is provided to process Shack-Hartmann and Digital Holography sensor data from the AEL:
```
python -m src.dataprocessing_ael [sh_sh/dh_dh/sh_dh]
```

## machine learning

The machine learning sensorfusion scripts may be run directly like
```
python -m src.fusion1d
```
Again, run with the `--help` flag to see information about command line parameters.

Alternatively, use the hyperparameter optimization tool [guild](https://guild.ai) to invoke runs and track results.
```
guild operations
guild run ks:train
guild run ks:train l2='[32,64,128]'
guild compare
guild compare -Fo ks:train --min loss
guild tensorboard
```