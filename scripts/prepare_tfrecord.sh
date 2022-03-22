#!/bin/bash

#bsub -n 4 -W 24:00 -N -R "rusage[ngpus_excl_p=1]" < ./prepare_tfrecord.sh

AUDIO_FILEPATTERN=data/audio/violin/*.mp3
TRAIN_TFRECORD=data/data.tfrecord

module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg

source venv/bin/activate

ddsp_prepare_tfrecord \
    --input_audio_filepatterns=$AUDIO_FILEPATTERN \
    --output_tfrecord_path=$TRAIN_TFRECORD \
    --num_shards=10 \
    --alsologtostderr
