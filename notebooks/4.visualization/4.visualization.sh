#!/bin/bash

# exit on error
# set -e

# install cell-injury-r environment
conda env create -f r_env.yaml

# activate conda env
conda activate cell-injury-r

# convert notebooks into python scripts
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run the scripts
Rscript nbconverted/4.visualization.r
