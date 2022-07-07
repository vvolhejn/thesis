TRAIN_TFRECORD_FILEPATTERN='/Users/vaclav/prog/thesis/data/datasets/violin3/violin3.tfrecord-train*'
SAVE_DIR='/Users/vaclav/prog/thesis/data/models/0703-rave-jukebox'
#GIN_FILE='gin/ddspae-jukebox.gin'
GIN_FILE='gin/rave-jukebox.gin'

nas_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="$SAVE_DIR" \
  --allow_memory_growth \
  --gin_search_path=/Users/vaclav/prog/thesis/gin/ \
  --gin_file="$GIN_FILE" \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'" \
  --gin_param="TFRecordProvider.with_jukebox=True" \
  --gin_param="TFRecordProvider.centered=True" \
  --gin_param="TFRecordProvider.frame_rate=50" \
  --gin_param="batch_size=8" \
  --gin_param="checkpoints_to_keep=5" \
  --gin_param="train_util.train.num_steps=30000" \
  --gin_param="train_util.train.steps_per_save=300" \
  --gin_param="train_util.train.steps_per_summary=300"