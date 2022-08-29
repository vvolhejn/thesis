# Accelerating Neural Audio Synthesis

This is the code accompanying my Master's thesis at ETH Zürich titled _Accelerating Neural Audio Synthesis_. The goal was to create a fast model for synthesizing musical audio in real time, based on [DDSP](https://arxiv.org/abs/2001.04643) and [RAVE](https://arxiv.org/abs/2111.05011). The resulting DDSP-based model, DDSP-CNN-Tiny, has less than 2500 parameters and runs at over 800x real-time on a CPU, while maintaining the quality of the original DDSP-full with 6M parameters.

Audio examples for the different models are available [here](https://vvolhejn.github.io/thesis/).

This is research code that needs modifying to be reusable (there are some hardcoded file paths, for instance), and active maintenance is not planned. Nevertheless, parts of the code can be useful for others:

- The code to define the DDSP-CNN-Tiny model is in [ddspae-cnn.gin](gin/ddspae-cnn.gin) (but set `CustomDilatedConvDecoder.casual=True` and `CustomDilatedConvDecoder.ch=8`) and [dilated_conv.py](thesis/dilated_conv.py)
- [thesis/runtimes/](thesis/runtimes/) contains code to convert models from TensorFlow, PyTorch or ONNX to various deep learning runtime libraries: TFLite, TorchScript, ONNX Runtime, OpenVINO, TVM and DeepSparse. Where applicable, it also includes code for quantizing the models through static or dynamic quantization.
- [thesis/](thesis/) includes a TensorFlow+DDSP re-implementation of components of [NEWT](https://arxiv.org/abs/2107.05050) and [RAVE](https://arxiv.org/abs/2111.05011), two models that were originally in PyTorch. There are modules such as [PQMF analysis and synthesis](thesis/pqmf.py) and [learnable waveshapers](thesis/newt.py).
- The code depends on a [modified fork of the DDSP library](https://github.com/vvolhejn/ddsp), which includes Weights and Biases integration and various other changes that couldn't be done without modifying the library.
- Notebooks to produce the figures seen in the thesis: [thesis-runtimes-plots.ipynb](notebooks/thesis-runtimes-plots.ipynb), [thesis-experiments3-plots.ipynb](notebooks/thesis-experiments3-plots.ipynb) and [survey-evaluation.ipynb](notebooks/survey-evaluation.ipynb).
- [A notebook](notebooks/download-samples.ipynb) to prepare the audio examples for GitHub pages.

## Training DDSP-CNN-Tiny

Since this was the most successful model, we include here the exact command to train it:

```bash
nas_prepare_job \
  -g ddspae-cnn.gin \
  -p  train_util.train.num_steps=100000 \ 
      TFRecordProvider.with_jukebox=False \
      TFRecordProvider.centered=True \
      TFRecordProvider.frame_rate=50 \
      CustomDilatedConvDecoder.causal=True \
      CustomDilatedConvDecoder.ch=8 \
  -d /users/vvolhejn/datasets/violin4/'*'.tfrecord-train'*'
```
This generates a script that is then submitted to SLURM via `sbatch`.
The `-p` argument overrides Gin parameters given in `ddspae-cnn.gin` and some implicitly loaded Gin files. The `TFRecordProvider` properties specify some dataset metadata (e.g. pitch/loudness frames per second), because this metadata is unfortunately not specified in the `.tfrecord` files used by DDSP. The `CustomDilatedConvDecoder` modify the decoder architecture.

The generated script (slightly cleaned up) is:
```bash
#!/bin/bash
# 0829-ddspae-cnn
#SBATCH --job-name=0829-ddspae-cnn
#SBATCH --time=16:00:00
#SBATCH --partition=amdrtx
#SBATCH --constraint=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --account=vvolhejn
#SBATCH --output=/users/vvolhejn/slurm-%j.out

source ~/.bashrc

conda activate nas

export CUDA_VISIBLE_DEVICES="0"

# This is some workaround to get CUDA working via Conda
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/users/vvolhejn/miniconda3/envs/nas/lib"

nvidia-smi

# 0829-ddspae-cnn is an automatically generated name, the initial
# digits determining the date (Aug 29) on which the model was created
mkdir -p /users/vvolhejn/models/0829-ddspae-cnn

wandb enabled
SAVE_DIR=/users/vvolhejn/models/0829-ddspae-cnn
TRAIN_TFRECORD_FILEPATTERN=/users/vvolhejn/datasets/violin4/*.tfrecord-train*

srun nas_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="$SAVE_DIR" \
  --allow_memory_growth \
  --gin_search_path=/users/vvolhejn/thesis/gin/ \
  --gin_file=ddspae-cnn.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'" \
  --gin_param="batch_size=8" \
  --gin_param="checkpoints_to_keep=1" \
  --gin_param="train_util.train.num_steps=100000" \
  --gin_param="TFRecordProvider.with_jukebox=False" \
  --gin_param="TFRecordProvider.centered=True" \
  --gin_param="TFRecordProvider.frame_rate=50" \
  --gin_param="CustomDilatedConvDecoder.causal=True" \
  --gin_param="CustomDilatedConvDecoder.ch=8" \
  --gin_param="train_util.train.steps_per_save=1000" \
  --gin_param="train_util.train.steps_per_summary=1000" \
  --gin_param="F0LoudnessPreprocessor.compute_f0=False" \
  --gin_param="OnlineF0PowerPreprocessor.compute_f0=False"
```

The model can then be evaluated using [`scripts/evaluate_one.sh`](scripts/evaluate_one.sh) via
```bash
scripts/evaluate_one.sh violin4:latest 0829-ddspae-cnn
```
Here the dataset to evaluate on is loaded from a W&B artifact - see section below on how to create one.

## Dataset creation

To turn audio files into a `.tfrecord` dataset usable by DDSP, use `ddsp_prepare_tfrecord`: see [`scripts/prepare_tfrecord.sh`](scripts/prepare_tfrecord.sh) or [`scripts/prepare_tfrecord_slurm.sh`](scripts/prepare_tfrecord_slurm.sh) for usage examples.

For timbre transfer, you need dataset statistics to match the loudness and pitch (octave) of the source audio to the distribution the model was trained on. See [`scripts/get_dataset_statistics.sh`](scripts/get_dataset_statistics.sh).

Finally, the dataset can be uploaded to a W&B artifact using [`scripts/create_wandb_dataset.py`](scripts/create_wandb_dataset.py). The evaluation script mentioned above works with datasets from W&B, but the older training script still uses `.tfrecord` files directly -- sorry!