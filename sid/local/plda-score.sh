#!/bin/bash


if [ $# -lt 1 ]; then
    echo "You need to enter the directory where the xvectors"
    echo "as well as the trained plda data can be found in."
    echo
    echo "Also you need to enter the path to where the eer file should be saved."
    echo
    exit 1
fi


trials=$TRIALS_FILE
xvector_dir=$1
eer_file=$2

calculate_score() {
    num=$1
    eer_file=$2
    plda=$xvector_dir/xvector_train_$num.plda
    mean=$xvector_dir/xvector_train_$num-mean.vec
    test_ark=$xvector_dir/xvector_test_$num.ark
    transf=$xvector_dir/xvector_train_$num-transform.mat

    xfname=$(echo $test_ark | sed 's/.ark//g')

    #STAGE1: generate scp
    mv $test_ark $xfname-tmp.ark
    copy-vector ark:$xfname-tmp.ark ark,scp:$test_ark,$xfname.scp
    scp=$xvector_dir/xvector_test_$num.scp
    rm $xfname-tmp.ark

    #STAGE2: run plda scoring
    ivector-plda-scoring --normalize-length=true \
        "ivector-copy-plda --smoothing=0.0 $plda - |" \
        "ark:ivector-subtract-global-mean $mean scp:$scp ark:- | transform-vec $transf ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $mean scp:$scp ark:- | transform-vec $transf ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$trials' | cut -d\  --fields=1,2 |" \
        $xfname.scores
    score=$xfname.scores

    #STAGE3: calculate eer
    eer=$(compute-eer <(python3 ./local/prepare_for_eer.py $trials $score))
    echo "eer of epoch $num is $eer" >> $eer_file
}

if [ $# -le 0 ]; then
    echo "You need to enter the file name of the score log file"
fi

epoch_count=$(ls $xvector_dir/*.plda | wc -l)
for I in $(seq 1 $epoch_count)
do
    let "num = $I - 1"
    mnum=$(printf "%02d" $num) 
    calculate_score $mnum $eer_file
done
