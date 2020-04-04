#!/bin/bash

# this script will extract xvectors on different GPUs in parallel in order
# to fully drive the whole machine if necesary

if [ $# -lt 6 ]; then
    echo "First you need to enter weather you want to extracto or validate."
    echo "Then you need to enter the flavour of the net,"
    echo "the dierectory to the models and the directory"
    echo "and the directory where the xvectors should be saved."
    echo "Also you need to enter how much devices you want to utalize"
    echo "and which is the first device."
    echo "run-parralel.sh extract <stat/attent> <models_dir> <xvectors_dir> <num_devices> <device_offset>"
    echo "or"
    echo "run-parralel.sh validate <stat/attent> <models_dir> <scores_dir> <num_devices> <device_offset>"
    exit 1
fi

train_scp=$TRAIN_SCP
test_scp=$TEST_SCP
operation=$1
flavour=$2
models_dir=$3
output_dir=$4 # for x-vector or scores
device_count=$5
offset_device=$6

models_count=$(ls $models_dir | wc -l)
tmpdir=.tmp/schabi$(date +%s)

exec_parralel() {
    let "dev=$I - 1 + $offset_device"
    for I in $(seq $1 $2 $3)
    do
        let "mnum = $I - 1"
        mnum=$(printf "%02d" $mnum)
        model=$models_dir/raw_model$mnum.pt
        echo "run parralel: start $mnum"

        # PyTorch prevents running models on cards other than thouse they where trained on.
        # This is because the .pt files are branded with the device of their training.
        # If we want to run the same model on different devices we need to "override" that branding.
        # DON'T TRY THIS AT HOME
        # Because of this models can not be trained on more then 10 cards in parralel.
        original_device=$(strings $model | grep cuda | head -c 6)
        cat $model | sed "s/$original_device/cuda:$dev/g" > $tmpdir/raw_model$mnum.pt
        
        if [ "$operation" = "extract" ]; then
            python3 src/extract.py $flavour $tmpdir/raw_model$mnum.pt $train_scp $output_dir/xvector_train_$mnum.ark cuda:$dev
            python3 src/extract.py $flavour $tmpdir/raw_model$mnum.pt $test_scp $output_dir/xvector_test_$mnum.ark cuda:$dev
        else
            python3 src/validate.py $flavour $tmpdir/raw_model$mnum.pt $output_dir/epoch_$mnum.score cuda:$dev
        fi
    done
}

mkdir $tmpdir
for I in $(seq 1 1 $device_count)
do
    exec_parralel $I $device_count $models_count &
done

echo "running"
wait
rm -r $tmpdir
echo "done"
