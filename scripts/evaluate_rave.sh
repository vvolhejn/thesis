#0324-fullrave-noiseless-2

#SAVE_DIR=/cluster/scratch/vvolhejn/models/0324-fullrave-noiseless-2
SAVE_DIR=/cluster/scratch/vvolhejn/models/0325-solo_instrument
TRAIN_TFRECORD_FILEPATTERN='/cluster/home/vvolhejn/datasets/violin/violin.tfrecord*'

module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg eth_proxy cuda/11.1.1 cudnn/8.1.0.77

source venv/bin/activate

nvidia-smi
lscpu

#ddsp_run \
#  --run_once \
#  --mode=eval \
#  --alsologtostderr \
#  --save_dir="$SAVE_DIR" \
#  --allow_memory_growth \
#  --gin_search_path=/cluster/home/vvolhejn/thesis/gin/ \
#  --gin_file=models/solo_instrument.gin \
#  --gin_file=datasets/tfrecord.gin \
#  --gin_param="TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'" \
#  --gin_param="eval_util.evaluate.batch_size=1" \
#  --gin_param="eval_util.evaluate.num_batches=100"
##  --gin_param="eval_util.evaluate_or_sample.evaluate_and_sample=True"
##  --gin_param="eval_util.evaluate.evaluate_and_sample=True"

nas_run \
  --alsologtostderr \
  --mode=eval \
  --save_dir="$SAVE_DIR" \
  --allow_memory_growth \
  --gin_search_path=/cluster/home/vvolhejn/thesis/gin/ \
  --gin_param="TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'" \
  --gin_param="eval_util.evaluate.batch_size=1" \
  --gin_param="eval_util.evaluate.num_batches=100"
#  --gin_param="eval_util.evaluate_or_sample.evaluate_and_sample=True"
#  --gin_param="eval_util.evaluate.evaluate_and_sample=True"
#  --gin_file=models/solo_instrument.gin \
#  --gin_file=datasets/tfrecord.gin \
#  --gin_file="${SAVE_DIR}/operative_config-0.gin" \