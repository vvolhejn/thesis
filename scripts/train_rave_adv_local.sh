#!/bin/bash



SAVE_DIR=~/thesis/data/models/0411-raveadv
TRAIN_TFRECORD_FILEPATTERN=$HOME'/prog/thesis/data/violin/violin.tfrecord*'
#bar

#module load gcc/8.2.0
#module load python_gpu/3.8.5
#module load libsndfile ffmpeg

#source venv/bin/activate

#nvidia-smi

ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="$SAVE_DIR" \
  --gin_search_path=/Users/vaclav/prog/thesis/gin/ \
  --gin_file=rave-adv.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'" \
  --gin_param="batch_size=2" \
  --gin_param="train_util.train.num_steps=1000" \
  --gin_param="train_util.train.steps_per_save=100" \
  --gin_param="train_util.train.steps_per_summary=100" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=2"
