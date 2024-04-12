#!/bin/bash

# activate conda env
conda activate cell-injury

# convert notebooks into python scripts
jupyter nbconvert --to python --output-dir=nbconverted/ *.ipynb

# run the scripts
python nbconverted/3.jump_analysis.py
python nbconverted/3.1.overlapping_compounds.py
