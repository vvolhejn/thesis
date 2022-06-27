#!/bin/bash
DIR=/home/vaclav/models_for_tvm

for f in "$DIR"/*.onnx; do
  without_ext="${f%.*}"
  echo Tuning $f

  tvmc tune \
    --target "llvm -mcpu=skylake-avx512" \
    --output "$without_ext"_number=1500.json \
    --number 250 \
    $f

  tvmc tune \
    --target "llvm -mcpu=skylake-avx512" \
    --output "$without_ext"_number=1500.json \
    --number 1500 \
    $f
done