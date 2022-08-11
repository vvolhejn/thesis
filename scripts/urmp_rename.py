# gsutil -m cp gs://magentadata/datasets/urmp/urmp_20210324/urmp_tpt_solo_ddsp_conditioning_test_batched'*' .
# https://console.cloud.google.com/storage/browser/magentadata/datasets/urmp/urmp_20210324;tab=objects?prefix=urmp_tpt_solo_ddsp_conditioning_train_batched&forceOnObjectsSortingFiltering=false&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22urmp_tpt_solo_ddsp_conditioning_train_batched_5C_22_22%257D%255D%22))

import os
import shutil
import re

for filename in os.listdir("."):
    if filename.startswith("urmp_") and ".tfrecord" in filename:
        match = re.match(
            r"urmp_(.*)_solo_ddsp_conditioning_(.*)_batched.tfrecord(.*)", filename
        )
        if match:
            # print(match.groups())
            instrument, test_or_train, index = match.groups()
            test_or_train = test_or_train.replace("test", "eval")
            new_name = "urmp_{}.tfrecord-{}{}".format(instrument, test_or_train, index)
            print(filename, new_name)
            shutil.move(filename, new_name)
        else:
            print("File not matched:", filename)
