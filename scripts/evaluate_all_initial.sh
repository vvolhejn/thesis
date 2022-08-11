#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
script=$SCRIPT_DIR/evaluate_one.sh

$script violin4:latest 0725-ddspae-2
$script violin4:latest 0715-ddspae-tiny
$script violin4:latest 0725-ddspae-cnn-1
$script violin4:latest 0726-fullrave-noiseless
$script violin4:latest 0726-ddspae-cnn
#$script violin4:latest 0725-newt
$script urmp_tpt2:latest 0805-ddspae
$script urmp_tpt2:latest 0805-ddspae-tiny
#$script urmp_tpt2:latest 0804-ddspae-cnn-3
#$script urmp_tpt2:latest 0809-ddspae-cnn
$script urmp_tpt2:latest 0805-newt