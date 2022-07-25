import argparse
import os

import wandb


def main(dataset_dir, wandb_name, filter_files=True):
    def keep(f):
        if filter_files:
            return ".tfrecord" in f or f.endswith(".md") or f.endswith(".sh")
        else:
            return True

    files = [f for f in os.listdir(dataset_dir) if keep(f)]
    assert files, f"No .tfrecord files found in {dataset_dir}"

    run = wandb.init(
        project="neural-audio-synthesis-thesis",
        entity="neural-audio-synthesis-thesis",
        name=f"dataset-{wandb_name}",
        # sync_tensorboard=True,
        config={
            "dataset_dir": dataset_dir,
            "wandb_name": wandb_name,
        },
        tags=["dataset"],
    )

    metadata = dict(
        example_secs=4,
        sample_rate=16000,
        frame_rate=50,
        centered=True,
        with_jukebox=True,
    )

    artifact = wandb.Artifact(wandb_name, type="dataset", metadata=metadata)

    for f in files:
        artifact.add_file(os.path.join(dataset_dir, f))

    run.log_artifact(artifact)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--all-files", action="store_true", dest="all_files")
    args = parser.parse_args()

    main(args.dataset_dir, args.wandb_name, filter_files=not args.all_files)
