SAVE_DIR=~/prog/thesis/data/models/0404-newt
TRAIN_TFRECORD_FILEPATTERN=$HOME'/prog/thesis/data/violin/violin.tfrecord*'

nas_run \
  --mode=eval \
  --alsologtostderr \
  --save_dir="$SAVE_DIR" \
  --allow_memory_growth \
  --gin_search_path=/Users/vaclav/prog/thesis/gin/ \
  --gin_param="TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'" \
  --gin_param="compute_f0.crepe_model='tiny'" \
  --gin_param="F0LoudnessPreprocessor.compute_f0=True"
