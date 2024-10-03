#!/bin/bash

# activate conda env
conda activate cell-injury

# convert notebooks into python scripts
jupyter nbconvert --to python --output-dir=nbconverted/ *.ipynb

# run the scripts
python nbconverted/2.1_fs_modeling.py
python nbconverted/2.2_aligned_modeling.py
