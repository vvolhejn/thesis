#eval all

module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg eth_proxy cuda/11.1.1 cudnn/8.1.0.77

source venv/bin/activate

# Print machine info
nvidia-smi
#lscpu
#hostnamectl

# 0324-halfrave-noiseless-4
for name in 0324-fullrave-noiseless-1 0404-newt 0328-ddspae-cnn 0328-ddspae; do
  SAVE_DIR=/cluster/scratch/vvolhejn/models/$name

  nas_run \
    --mode=eval \
    --alsologtostderr \
    --save_dir="$SAVE_DIR" \
    --allow_memory_growth \
    --gin_search_path=/cluster/home/vvolhejn/thesis/gin/ \
    --gin_param="batch_size=1" \
    --gin_param="F0LoudnessPreprocessor.compute_f0=True" \
    --gin_param="compute_f0.model_name='spice-v2'" \
    --wandb_group="0408-eval-2z"
done
