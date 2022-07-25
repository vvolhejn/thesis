#!/usr/local/bin/bash
#SBATCH --time=08:00:00
#SBATCH --partition=amdv100,intelv100,amdrtx,amda100
#SBATCH --constraint=gpu
#SBATCH --account=vvolhejn
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

echo "Running"

nvidia-smi

source ~/.bashrc

conda activate nas

echo Python executable: "$(which python || true)"

export CUDA_VISIBLE_DEVICES="0"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/users/vvolhejn/miniconda3/envs/nas/lib"


AUDIO_FILEPATTERN="$HOME/datasets_raw/guitarset/*solo_mix.wav"
TRAIN_TFRECORD=$HOME'/datasets/guitar4/guitar4.tfrecord'
#AUDIO_FILEPATTERN="$HOME/datasets_raw/violin/*.mp3"
#TRAIN_TFRECORD=$HOME'/datasets/violin5/violin5.tfrecord'

ddsp_prepare_tfrecord \
    --input_audio_filepatterns=$AUDIO_FILEPATTERN \
    --output_tfrecord_path=$TRAIN_TFRECORD \
    --sample_rate=44100 \
    --num_shards=10 \
    --alsologtostderr \
    --frame_rate=50 \
    --example_secs=4.0 \
    --hop_secs=1.0 \
    --viterbi=True \
    --jukebox=True \
    --center=True \
    --eval_split_fraction=0.1
