#!/bin/bash
# prepare tfrecord
#bsub -n 4 -W 24:00 -N -R "rusage[ngpus_excl_p=1]" < ./prepare_tfrecord.sh

AUDIO_FILEPATTERN="$SCRATCH/data/audio/transfer2/*.wav"
TRAIN_TFRECORD='/cluster/home/vvolhejn/datasets/transfer4/transfer4.tfrecord'

module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg eth_proxy cuda/11.1.1 cudnn/8.1.0.77

source venv/bin/activate

ddsp_prepare_tfrecord \
    --input_audio_filepatterns=$AUDIO_FILEPATTERN \
    --output_tfrecord_path=$TRAIN_TFRECORD \
    --num_shards=10 \
    --alsologtostderr \
    --frame_rate=50 \
    --example_secs=4.0 \
    --hop_secs=1.0 \
    --viterbi=True \
    --center=True
