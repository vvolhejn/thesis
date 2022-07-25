"""
This didn't work at first, I submitted an issue about this in the DDSP repo:
https://github.com/magenta/ddsp/issues/427

Make sure the TFRecordProvider arguments match what the dataset was created with.
"""

import argparse
import os

import pickle
import ddsp.training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_tfrecord_filepattern", help="Glob of the .tfrecord files to analyze"
    )
    parser.add_argument("save_dir", help="In which directory to save the statistics")
    args = parser.parse_args()

    data_provider = ddsp.training.data.TFRecordProvider(
        args.train_tfrecord_filepattern,
        # Make sure these arguments match what the dataset was created with!
        # sample_rate=44100,
        frame_rate=50,
        centered=True,
        with_jukebox=False,
    )
    dataset = data_provider.get_dataset(shuffle=False)

    filename = "dataset_statistics.pkl"
    pickle_path = os.path.join(args.save_dir, filename)

    if os.path.exists(pickle_path):
        raise ValueError(f"The file {pickle_path} already exists.")

    ds_stats = ddsp.training.postprocessing.compute_dataset_statistics(
        data_provider, batch_size=1
    )

    with open(pickle_path, 'wb') as f:
        pickle.dump(ds_stats, f)

    print(f'Done! Saved dataset statistics to: {pickle_path}')

