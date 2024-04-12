#!/bin/bash

# activate conda env
conda activate cell-injury

# convert notebooks into python scripts
jupyter nbconvert --to python --output-dir=nbconverted/ *.ipynb

# run the scripts
python nbconverted/2.modeling.py
