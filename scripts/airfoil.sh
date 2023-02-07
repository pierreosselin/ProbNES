#!/bin/sh

wget -O ./data/airfoil_self_noise.dat https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat

python ./scripts/airfoil_processing.py