"""
Prepare a job for LSF's `bsub`.
The idea is to run this and pipe the output to `bsub`, e.g.
$ prepare_job -g fullrave.gin -p train_util.train.num_steps=100000 \
    | bsub -n 4 -W 24:00 -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]"
"""

import argparse
import datetime
import os

from thesis.prepare_job_util import get_today_string, add_distinguishing_suffix

HEADER = r"""
module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg eth_proxy cuda/11.1.1 cudnn/8.1.0.77

source venv/bin/activate

# Print machine info
nvidia-smi
#lscpu
#hostnamectl

"""

COMMAND_TEMPLATE = r"""
SAVE_DIR={base_dir}/{job_dir}
TRAIN_TFRECORD_FILEPATTERN={dataset_pattern} 

{wandb_command}

nas_run \
  --mode={mode} \
  --alsologtostderr \
  --save_dir="$SAVE_DIR" \
  --allow_memory_growth \
  --gin_search_path=/cluster/home/vvolhejn/thesis/gin/ \
  {mode_specific_params} \
  {gin_params}
"""

#   --gin_param="train_util.train.num_steps=2000000" \
#   --gin_param="train_util.train.steps_per_save=10000" \

DEFAULT_GIN_PARAMS = {
    "batch_size": "8",
    # "trainers.Trainer.checkpoints_to_keep": "5",
    "checkpoints_to_keep": "5",
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
):

    job = f"#{comment}\n"
    job += HEADER

    job += "wandb enabled\n" if use_wandb else "wandb disabled\n"

    job += f"SAVE_DIR={base_dir}/{job_dir}\n"
    job += f"TRAIN_TFRECORD_FILEPATTERN={dataset_pattern}\n"
    job += "\n"

    params = [
        f"--mode={mode}",
        "--alsologtostderr",
        '--save_dir="$SAVE_DIR"',
        "--allow_memory_growth",
        "--gin_search_path=/cluster/home/vvolhejn/thesis/gin/",
    ]

    if use_runtime:
        params.append("--use_runtime")

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

    job += " \\\n  ".join(["nas_run"] + params)

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
        default="/cluster/scratch/vvolhejn/models",
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
        default="'/cluster/home/vvolhejn/datasets/violin2/violin2.tfrecord-train*'",
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
    )
    print(job)


if __name__ == "__main__":
    console_entry_point()
