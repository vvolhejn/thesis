#!/bin/bash
# Kill on first error
set -e

# Dry run
wandb disabled
python3 -m thesis.benchmark --n-layers 1 --n-sizes 1 --n-iterations 1 --kind dilated_cnn
python3 -m thesis.benchmark --n-layers 1 --n-sizes 1 --n-iterations 1 --kind dilated_cnn_ib
python3 -m thesis.benchmark --n-layers 1 --n-sizes 1 --n-iterations 1 --kind inverted_bottleneck
python3 -m thesis.benchmark --n-layers 2 --n-sizes 1 --n-iterations 1 --kind dense

wandb enabled
python3 -m thesis.benchmark --n-layers 3 --n-sizes 2 --n-iterations 50 --kind dilated_cnn
python3 -m thesis.benchmark --n-layers 3 --n-sizes 2 --n-iterations 50 --kind dilated_cnn_ib
python3 -m thesis.benchmark --n-layers 1 --n-sizes 8 --n-iterations 50 --kind inverted_bottleneck
python3 -m thesis.benchmark --n-layers 4 --n-sizes 4 --n-iterations 100 --kind dense