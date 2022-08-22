import argparse
import os
import datetime

import pandas as pd
import wandb


def main(path):
    run = wandb.init(
        project="neural-audio-synthesis-thesis",
        entity="neural-audio-synthesis-thesis",
        name=f"backup-{datetime.datetime.now().strftime('%y-%d-%m-%H-%M')}",
        # sync_tensorboard=True,
        config={
            "path": path,
        },
        tags=["survey-backup"],
    )

    artifact = wandb.Artifact("survey-backup", type="survey-backup", metadata={"path": path})
    artifact.add_file(path)
    run.log_artifact(artifact)

    df = pd.read_csv(path)
    run.log({"survey-results": wandb.Table(dataframe=df)})

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    main(args.path)
