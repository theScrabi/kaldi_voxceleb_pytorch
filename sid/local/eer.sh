#!/bin/bash

source ../path.sh
if [ $# -lt 2 ]; then
    echo "You need to enter the target log file, "
    echo "as well as the directory where the score files can be found."
    exit 1
fi

eer_file=$1
score_dir=$2

count=$(ls $score_dir/*.score | wc -l) 
let count=$count-1

for I in $(seq 0 $count)
do
    mnum=$(printf "%02d" $I)
    score_file="$score_dir/epoch_$mnum.score"
    echo "eer of epoch $I is $($KALDI/ivectorbin/compute-eer $score_file 2>> exp/log )" >> $eer_file
done

