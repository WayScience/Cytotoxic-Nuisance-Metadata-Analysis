#!/bin/bash

# exit on error
# set -e

# activate conda env
conda activate cell-injury

# convert notebooks into python scripts
jupyter nbconvert --to python --output-dir=nbconverted/ *.ipynb

# run the scripts
python nbconverted/0.feature_selection.py
