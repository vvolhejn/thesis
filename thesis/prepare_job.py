"""
Prepare a job for LSF's `bsub`.
The idea is to run this and pipe the output to `bsub`, e.g.
$ prepare_job -g fullrave.gin -p train_util.train.num_steps=100000 \
    | bsub -n 4 -W 24:00 -R "rusage[mem=8192]" -R "rusage[ngpus_excl_p=1]"
"""

import argparse
import datetime
import os

template = r"""#{job_dir}

SAVE_DIR={base_dir}/{job_dir}
TRAIN_TFRECORD_FILEPATTERN={dataset_pattern} 

module load gcc/8.2.0
module load python_gpu/3.8.5
module load libsndfile ffmpeg eth_proxy cuda/11.1.1 cudnn/8.1.0.77

source venv/bin/activate

nvidia-smi

ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="$SAVE_DIR" \
  --allow_memory_growth \
  --gin_search_path=/cluster/home/vvolhejn/thesis/gin/ \
  --gin_file={gin_file} \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='$TRAIN_TFRECORD_FILEPATTERN'" \
  {gin_params}
"""
#   --gin_param="train_util.train.num_steps=2000000" \
#   --gin_param="train_util.train.steps_per_save=10000" \

DEFAULT_GIN_PARAMS = {
    "batch_size": "8",
    "trainers.Trainer.checkpoints_to_keep": "5",
}


class ParseGinParams(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, DEFAULT_GIN_PARAMS.copy())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def prepare_job(base_dir, job_dir, dataset_pattern, gin_file, gin_params):
    gin_params_s = " \\\n  ".join(
        f'--gin_param="{k}={v}"' for (k, v) in gin_params.items()
    )

    job = template.format(
        base_dir=base_dir,
        job_dir=job_dir,
        dataset_pattern=dataset_pattern,
        gin_file=gin_file,
        gin_params=gin_params_s,
    )

    return job


def add_distinguishing_suffix(base_dir, name):
    """
    Choose a dir name that doesn't exist yet by trying to append '-1', '-2' and so on.
    """
    n = 0
    while True:
        candidate = name
        if n > 0:
            candidate += f"-{n}"

        path = os.path.expanduser(os.path.join(base_dir, candidate))
        if not os.path.exists(path):
            return candidate
        elif os.path.isdir(path) and os.listdir(path) == []:
            # If the directory exists but is empty, this probably means an earlier
            # attempt at submission crashed. Reuse the directory in this case.
            return candidate
        else:
            n += 1


def create_job_dir(base_dir, name, gin_file):
    gin_file = os.path.basename(gin_file)
    assert gin_file.endswith(".gin")
    gin_file = gin_file[: -len(".gin")]

    today = datetime.date.today()
    date_s = today.strftime("%m%d")

    res = f"{date_s}-{gin_file}"
    if name:
        res += f"-{name}"

    res = add_distinguishing_suffix(base_dir, res)

    return res


def console_entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base-dir",
        default="~/data",
        help="Directory under which the models are saved",
    )
    parser.add_argument("-n", "--name", help="Human-readable identifier suffix")
    parser.add_argument(
        "-d",
        "--dataset",
        default="$HOME'/data/data.tfrecord*'",
        help="An absolute glob pattern of .tfrecord files to use",
    )
    parser.add_argument(
        "-g",
        "--gin-file",
        required=True,
        help="Filename of the .gin file describing the model (e.g. 'newt.gin')",
    )
    parser.add_argument("-s", "--steps", help="Number of training steps", type=int)
    parser.add_argument(
        "-p",
        "--gin-params",
        nargs="*",
        action=ParseGinParams,
        help="Additional Gin parameters passed like key1=value key2=value2",
    )

    args = parser.parse_args()

    gin_params = args.gin_params or DEFAULT_GIN_PARAMS.copy()

    if "train_util.train.num_steps" not in gin_params:
        gin_params["train_util.train.num_steps"] = args.steps or 30000
    elif args.steps:
        raise ValueError(
            "Number of training steps set both through --steps and --gin-params"
        )

    if "train_util.train.steps_per_save" not in gin_params:
        # Do 100 saves during the training.
        steps_per_save = int(gin_params["train_util.train.num_steps"]) // 100
        assert steps_per_save > 0
        gin_params["train_util.train.steps_per_save"] = str(steps_per_save)

    job_dir = create_job_dir(args.base_dir, args.name, args.gin_file)

    job = prepare_job(args.base_dir, job_dir, args.dataset, args.gin_file, gin_params)
    print(job)


if __name__ == "__main__":
    console_entry_point()
