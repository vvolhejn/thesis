#!/bin/bash

#https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

script=$SCRIPT_DIR/evaluate_one.sh
names="0725-ddspae-cnn-1 0809-ddspae-cnn-5 0726-ddspae-cnn"
dataset=violin4

#names="0804-ddspae-cnn-3 0809-ddspae-cnn-4 0809-ddspae-cnn"
#dataset=urmp_tpt2

for name in $names; do
  echo "---------------- EVALUATING $name ----------------"
  #SAVE_DIR=/cluster/scratch/vvolhejn/models/$name

  for suffix in "--use_runtime" "--use_runtime --quantization"; do
#  for suffix in "--use_runtime" "'--use_runtime --quantization"; do
    echo $script $dataset:latest $name "$suffix"
    $script $dataset:latest $name "$suffix"
  done
done
