#!/bin/bash

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 dataset model_name suffix"
  echo "e.g. $0 guitar3:latest 0725-ddspae-cnn '--use_runtime --quantization'"
  exit 1
fi

dataset=$1
model_name=$2
suffix=$3

nas_run \
  --mode=eval \
  --alsologtostderr \
  --gin_search_path=/home/vaclav/thesis/gin/ \
  --gin_param="evaluate_or_sample_batch.recompute_f0=True" \
  --gin_param="nas_evaluate.num_batches=-1" \
  --gin_param="nas_evaluate.num_calibration_batches=500" \
  --gin_param="F0LoudnessPreprocessor.compute_f0=True" \
  --gin_param="F0LoudnessPreprocessor.frame_rate=250" \
  --gin_param="compute_f0.model_name='crepe-tiny'" \
  --gin_param="nas_evaluate.cache_newt_waveshapers=True" \
  --gin_param="Reverb.decay_after=16000" \
  --gin_param="WandbTFRecordProvider.artifact_name='neural-audio-synthesis-thesis/${dataset}'" \
  --save_dir="/home/vaclav/models/${model_name}" \
  $suffix