# Warning: broken (see get_dataset_statistics.py)
#bsub -n 4 -W 24:00 -N -R "rusage[ngpus_excl_p=1]" < ./train_ddsp.sh

SAVE_DIR=~/datasets/flute_dummy/
TRAIN_TFRECORD_FILEPATTERN=$HOME'/datasets/violin/violin.tfrecord*'
#bar

module load libsndfile ffmpeg gcc/8.2.0 python_gpu/3.8.5 cudnn/8.1.0.77

source venv/bin/activate

nvidia-smi

python3 ~/thesis/scripts/get_dataset_statistics.py "$TRAIN_TFRECORD_FILEPATTERN" "$SAVE_DIR"