#!/bin/bash

# activate conda env
conda activate cell-injury

# convert notebooks into python scripts
jupyter nbconvert --to python --output-dir=nbconverted/ *.ipynb

# run the scripts
python nbconverted/1.1_cell_injury_data_splits.py
python nbconverted/1.2_aligned_cell_injury_data_splits.py
