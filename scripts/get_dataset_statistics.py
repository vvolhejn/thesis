"""
Warning: this doesn't work.

I submitted an issue about this in the DDSP repo: https://github.com/magenta/ddsp/issues/427

Running it a month later, it fails with a different error (??), namely:

    Traceback (most recent call last):
      File "/cluster/home/vvolhejn/thesis/scripts/get_dataset_statistics.py", line 24, in <module>
        ds_stats = ddsp.training.postprocessing.compute_dataset_statistics(
      File "/cluster/home/vvolhejn/ddsp/ddsp/training/postprocessing.py", line 352, in compute_dataset_statistics
        ds_stats.update(get_stats(power_trimmed, 'power_note', mask_on))
      File "/cluster/home/vvolhejn/ddsp/ddsp/training/postprocessing.py", line 327, in get_stats
        max_list.append(np.max(x_i[m]))
    IndexError: boolean index did not match indexed array along dimension 0; dimension is 181 but corresponding boolean dimension is 980

My workaround was to copy the dataset statistics of the violin dataset from DDSP.
Obviously not scalable.
"""

import argparse
import os

import ddsp.training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_tfrecord_filepattern", help="Glob of the .tfrecord files to analyze"
    )
    parser.add_argument("save_dir", help="In which directory to save the statistics")
    args = parser.parse_args()

    data_provider = ddsp.training.data.TFRecordProvider(args.train_tfrecord_filepattern)
    dataset = data_provider.get_dataset(shuffle=False)

    filename = "dataset_statistics.pkl"
    pickle_path = os.path.join(args.save_dir, filename)

    if os.path.exists(pickle_path):
        raise ValueError(f"The file {pickle_path} already exists.")

    ds_stats = ddsp.training.postprocessing.compute_dataset_statistics(
        data_provider, batch_size=1
    )
