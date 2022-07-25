"""
Prepare a job for LSF's `bsub`.
The idea is to run this and pipe the output to `bsub`, e.g.
$ prepare_job -g fullrave.gin -p train_util.train.num_steps=100000 \
    | bsub -n 4 -W 24:00 -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]"
"""

import argparse
import datetime
import os
import shutil
import warnings

from thesis.prepare_job_util import get_today_string, add_distinguishing_suffix

HOME_DIR = os.path.expanduser("~")

LSF_HEADER = r"""
{comment}
module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg eth_proxy cuda/11.1.1 cudnn/8.1.0.77

source venv/bin/activate

# Print machine info
nvidia-smi
#lscpu
#hostnamectl

"""

SLURM_HEADER = r"""#!/bin/bash
# {comment}
#SBATCH --job-name={comment}
#SBATCH --time=8:00:00
#SBATCH --partition=amdrtx
#SBATCH --constraint=gpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --account=vvolhejn
# Unused:
#aSBATCH --output={save_dir}/slurm-%j.out
#aSBATCH --partition=amdv100,intelv100,amdrtx,amda100

source ~/.bashrc

conda activate nas

export CUDA_VISIBLE_DEVICES="{gpu_index}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/users/vvolhejn/miniconda3/envs/nas/lib"

nvidia-smi

mkdir -p {save_dir}

"""

# #SBATCH --job-name=job_name
# #SBATCH --time=01:00:00
# #SBATCH --nodes=2
# #SBATCH --ntasks-per-core=2
# #SBATCH --ntasks-per-node=12
# #SBATCH --cpus-per-task=2

#   --gin_param="train_util.train.num_steps=2000000" \
#   --gin_param="train_util.train.steps_per_save=10000" \

DEFAULT_GIN_PARAMS = {
    "batch_size": "8",
    # "trainers.Trainer.checkpoints_to_keep": "5",
    "checkpoints_to_keep": "1",
    # For evaluation
    # "compute_f0.model_name": "'spice-v2'",  # "'crepe-tiny'"
}


class ParseGinParams(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, DEFAULT_GIN_PARAMS.copy())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def prepare_job(
    mode,
    comment,
    base_dir,
    job_dir,
    dataset_pattern,
    gin_params,
    mode_specific_params,
    use_wandb,
    use_runtime,
    quantization,
    gpu_index,
):
    workload_manager = get_workload_manager()

    job = SLURM_HEADER.format(
        comment=comment, save_dir=os.path.join(base_dir, job_dir), gpu_index=gpu_index
    )

    job += "wandb enabled\n" if use_wandb else "wandb disabled\n"

    job += f"SAVE_DIR={base_dir}/{job_dir}\n"
    job += f"TRAIN_TFRECORD_FILEPATTERN={dataset_pattern}\n"
    job += "\n"

    params = [
        f"--mode={mode}",
        "--alsologtostderr",
        '--save_dir="$SAVE_DIR"',
        "--allow_memory_growth",
        f"--gin_search_path={HOME_DIR}/thesis/gin/",
    ]

    if use_runtime:
        params.append("--use_runtime")

    if quantization:
        params.append("--quantization")

    params += mode_specific_params

    # Fill in the parameters for the preprocessor that decide whether a cached f0
    # should be used (for training) or whether it should be computed anew (for inference
    # on unseen inputs and evaluation)
    for param_name in [
        "F0LoudnessPreprocessor.compute_f0",
        "OnlineF0PowerPreprocessor.compute_f0",
    ]:
        if param_name not in gin_params:
            gin_params[param_name] = False if (mode == "train") else True

    params += [f'--gin_param="{k}={v}"' for (k, v) in gin_params.items()]

    command_executable = "nas_run" if (workload_manager == "lsf") else "srun nas_run"

    job += " \\\n  ".join([command_executable] + params)

    return job


def create_job_dir(base_dir, name, gin_file):
    gin_file = os.path.basename(gin_file)
    assert gin_file.endswith(".gin")
    gin_file = gin_file[: -len(".gin")]

    res = f"{get_today_string()}-{gin_file}"
    if name:
        res += f"-{name}"

    res = add_distinguishing_suffix(base_dir, res)

    return res


def add_default_training_gin_params(gin_params, steps):
    if "train_util.train.num_steps" not in gin_params:
        gin_params["train_util.train.num_steps"] = steps or 30000
    elif steps:
        raise ValueError(
            "Number of training steps set both through --steps and --gin-params"
        )

    if "train_util.train.steps_per_save" not in gin_params:
        # Try to do 100 saves during the training
        # but do not save more often than once per 100 steps.
        steps_per_save = max(int(gin_params["train_util.train.num_steps"]) // 100, 100)
        assert steps_per_save > 0
        gin_params["train_util.train.steps_per_save"] = str(steps_per_save)

    if "train_util.train.steps_per_summary" not in gin_params:
        gin_params["train_util.train.steps_per_summary"] = gin_params[
            "train_util.train.steps_per_save"
        ]


def console_entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval"],
        help="Whether to train or eval",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        default=f"{HOME_DIR}/models",
        help="Directory under which the models are saved",
    )
    parser.add_argument(
        "-j",
        "--job-dir",
        help="For evaluation: the directory of the model to evaluate, "
        "relative to the base dir.",
    )
    parser.add_argument("-n", "--name", help="Human-readable identifier suffix")
    parser.add_argument(
        "-d",
        "--dataset",
        default=f"'{HOME_DIR}/datasets/violin2/violin2.tfrecord-train*'",
        help="An absolute glob pattern of .tfrecord files to use",
    )
    parser.add_argument(
        "-g",
        "--gin-file",
        help="For training: Filename of the .gin file "
        "describing the model (e.g. 'newt.gin')",
    )
    parser.add_argument("-s", "--steps", help="Number of training steps", type=int)
    parser.add_argument(
        "-p",
        "--gin-params",
        nargs="*",
        action=ParseGinParams,
        help="Additional Gin parameters passed like key1=value key2=value2",
    )
    parser.add_argument(
        "--no-wandb",
        help="Disable Weights and Biases for this run",
        dest="use_wandb",
        default=True,
        action="store_false",
    )
    parser.add_argument(
        "--use-runtime",
        help="Use a specialized runtime library to run the decoder for evaluation",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--quantization",
        help="Whether the runtime from --use-runtime should use quantization",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--gpu-index",
        help="Which GPU (0-indexed) to use",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    gin_params = args.gin_params or DEFAULT_GIN_PARAMS.copy()

    if args.mode == "train":
        assert args.gin_file is not None, "You must specify --gin-file when training."

        add_default_training_gin_params(gin_params, args.steps)

        job_dir = create_job_dir(args.base_dir, args.name, args.gin_file)

        mode_specific_params = [
            f"--gin_file={args.gin_file}",
            "--gin_file=datasets/tfrecord.gin",
            "--gin_param=\"TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'\"",
        ]
        comment = job_dir
    elif args.mode == "eval":
        mode_specific_params = []

        job_dir = args.job_dir
        assert job_dir is not None, "You must specify --job-dir when evaluating."

        comment = f"eval {job_dir}"
    else:
        raise ValueError(f"Unknown mode '{args.mode}'")

    job = prepare_job(
        args.mode,
        comment,
        args.base_dir,
        job_dir,
        args.dataset,
        gin_params,
        mode_specific_params,
        args.use_wandb,
        args.use_runtime,
        args.quantization,
        args.gpu_index,
    )
    print(job)


def get_workload_manager():
    if shutil.which("squeue"):
        return "slurm"
    elif shutil.which("bjobs"):
        return "lsf"
    else:
        warnings.warn("Unrecognized workload manager, defaulting to LSF")
        return "lsf"


if __name__ == "__main__":
    console_entry_point()
