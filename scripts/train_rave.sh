#!/bin/bash
#RAVE.
#bsub -n 4 -W 24:00 -N -R "rusage[ngpus_excl_p=1]" < ./train_newt.sh

SAVE_DIR=~/data/train_rave5
TRAIN_TFRECORD_FILEPATTERN=$HOME'/data/data.tfrecord*'
#bar

module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg eth_proxy cuda/11.1.1 cudnn/8.1.0.77

source venv/bin/activate

nvidia-smi

ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="$SAVE_DIR" \
  --allow_memory_growth \
  --gin_search_path=/cluster/home/vvolhejn/thesis/gin/ \
  --gin_file=rave.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'" \
  --gin_param="batch_size=8" \
  --gin_param="train_util.train.num_steps=2000000" \
  --gin_param="train_util.train.steps_per_save=10000" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=5"
