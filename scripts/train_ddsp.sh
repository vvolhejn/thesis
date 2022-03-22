#!/bin/bash
#DDSP.
#bsub -n 4 -W 24:00 -N -R "rusage[ngpus_excl_p=1]" < ./train_ddsp.sh

SAVE_DIR=~/data/train2
TRAIN_TFRECORD_FILEPATTERN=$HOME'/data/data.tfrecord*'
#bar

module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg eth_proxy cuda/11.1.1

source venv/bin/activate

nvidia-smi

ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="$SAVE_DIR" \
  --gin_file=models/solo_instrument.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'" \
  --gin_param="batch_size=8" \
  --gin_param="train_util.train.num_steps=30000" \
  --gin_param="train_util.train.steps_per_save=300" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=10"
