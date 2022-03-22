#!/bin/bash
#Test GPU.
module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg eth_proxy cuda/11.1.1

source venv/bin/activate

echo "hello"
which python3
python3 --version

python3 ~/thesis/scripts/test_gpu.py
