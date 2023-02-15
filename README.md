# sensorfusion via shallow decoder neural networks
This repository contains work demonstrating the fusion of a low-spatial resolution (but high-temporal resolution) sensor to a high-spatial resolution (but low-spatial resolution) sensor using a shallow decoder neural network map.

## data
Datasets used include a 1-D Kuramoto-Sivashinsky simulation as well as 2-D Shack-Hartmann and Digital Holography sensors for aero-optical metrology, experimental data provided by the Aero Effects Laboratory (AEL) at AFRL. AEL data is not currently provided, but code to generate 1-D Kuramoto-Sivashisnky (KS) equation data is provided under `data/ks/raw/ks.m`. Running this MATLAB file creates a 1.4G file with 98000 snapshots, each with 2048 spatial pixels.

### subsampling into training datasets
To process the KS superset into smaller datasets for experiments, run the provided dataprocessing script from the base directory. Use the `--help` flag for subsampling options. For example, the following creates a dataset where the low-spatial resolution sensor has been downsampled to 1/64 the spatial pixels and uses two temporal embeddings for each input, while the high-spatial resolution sensor is sampled at every 10th frame.
```
python -m src.dataprocessing_ks --xr 64 --en 2 --tr 10
```
A similar script is provided for AEL dataset users, where the sensor pairing needs to be declared as well.
```
python -m src.dataprocessing_ael [sh_sh | dh_dh | sh_dh] [options...]
```

## machine learning
The sensorfusion modules may be run directly from the base directory. Run with the `--help` flag to see information about command line parameters.
```
python -m src.fusion1d
```
You can also check out the provided Jupyter notebook for the KS (fusion1d) problem.

### guild (hyperparameter optimization)
Alternatively, use the hyperparameter optimization tool [guild](https://guild.ai) and the `guild.yml` file to invoke runs and track results. Here are some example queries.


> View all operations. Run a KS model with various hidden layer sizes. Run it again with specific layers and tune the learning rate with Bayesian optimization. Turn off video creation. View results for that model, sorted by loss. Then check out tensorboard.
```
guild operations
guild run ks:train l1='[512,2048]' l2='[32,64,128]'
guild run ks:train l1=2048 l2=32 lr='[0.01:0.1]' --optimizer gp MAKEVIDS=False
guild compare -Fo ks:train --min loss
guild tensorboard
```
