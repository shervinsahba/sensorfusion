# sensorfusion via shallow decoder neural networks

This repository contains work demonstrating the fusion of a low-spatial resolution (but high-temporal resolution) sensor to the inverse, a high-spatial resolution (but low-spatial resolution) sensor.

This stufy included 1-D Kuramoto-Sivashinsky simulations as well as 2-D Shack-Hartmann and Digital Holography sensors used in optical metrology at the Aero Effects Laboratory at AFRL. AEL data is not currently provided, but code is provided to generate a KS simulation dataset.


## data

Code to generate data using the 1-D Kuramoto-Sivashisnky equation is provided under `data/ks/ks.m`. Running this provides simulation data that can then be processed as you require. I suggest you place the generated file under `data/ks/raw/`.

From the root directory, run
```
python -m src.dataprocessing_ks
```
to subsample the ks data as desired. Run with the `--help` flag for subsampling and embedding options.

AEL data is not currently provided, but as a note for the future, it too can be subsampled with
```
python -m src.dataprocessing_ael [sh_sh/dh_dh/sh_dh]
```
where the required sensor pairings must be provided.
