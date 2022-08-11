#!/bin/bash

#https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

script=$SCRIPT_DIR/evaluate_one.sh
#names="0804-ddspae-cnn-4 0804-ddspae-cnn-7 0804-ddspae-cnn-8 0805-ddspae-cnn"
#names="0809-ddspae-cnn-3 0809-ddspae-cnn-4"
names="0726-ddspae-cnn 0809-ddspae-cnn-5 0809-ddspae-cnn-6 0809-ddspae-cnn-7"

for name in $names; do
  echo "---------------- EVALUATING $name ----------------"
  #SAVE_DIR=/cluster/scratch/vvolhejn/models/$name

  for suffix in "--use_runtime" "--use_runtime --quantization"; do
#  for suffix in "--use_runtime" "'--use_runtime --quantization"; do
    echo $script urmp_tpt2:latest $name "$suffix"
    $script urmp_tpt2:latest $name "$suffix"
  done
done
