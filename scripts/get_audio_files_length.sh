#!/bin/bash

LENGTH=0

for f in *.wav; do
  CUR=$(ffprobe -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $f 2>/dev/null)
  LENGTH="$LENGTH+$CUR"
done

#for f in *.mp3; do
#  CUR=$(ffprobe -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $f 2>/dev/null)
#  LENGTH="$LENGTH+$CUR"
#done

echo "Total length of files in $(pwd) in seconds:"
echo $LENGTH | bc
