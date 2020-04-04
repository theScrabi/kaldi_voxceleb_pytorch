#!/bin/bash

# This is a try to replicate the kaldi stages for this sid implementation.
# Additionally this script also introduces flavours. Please read the
# README.md to find out more about the flavour concept.
#
# This script excepcts that the MFCCs where already generated.

# stage depicts from which stage on the code should run.
# In the first run stage should be 0.
source path.sh

stage=0
logfile=exp/log

echo "" > $logfile

if [ $# -lt 3 ]; then
    echo
    echo "You need to enter which flavour of the net you want to run"
    echo "This is done with multiple parameters. First is"
    echo "attent for attention or stat for statistical"
    echo "Second parameter is:"
    echo "plda or nonplda"
    echo "third parameter is the name of the log file that should be generated"
    echo
    exit 1
fi

net_flavour=$1
after_flavour=$2
logfile_template=$3
flavour=$net_flavour-$after_flavour

#make directories
models_dir=exp/models/$flavour
xvectors_dir=exp/xvectors/$flavour
scores_dir=exp/scores/$flavour
eer_file=exp/eer/$logfile_template-$flavour-$(date +"%H:%M:%S_%d.%m.%Y").eer
loss_log=exp/logs/$logfile_template-$flavour-$(date +"%H:%M:%S_%d.%m.%Y")-train.log
mkdir -p $models_dir >> $logfile
mkdir -p $xvectors_dir >> $logfile
mkdir -p $scores_dir >> $logfile


# train 
if [ $stage -le 0 ]; then
    bash ./local/genlbl.sh > exp/spklbl
    echo "###################### train ######################" >> $logfile
    date >> $logfile
    python3 ./src/train.py $net_flavour $models_dir >> $logfile 2>&1 
    
    if [ $? -ne 0 ]; then
        echo "Quit due to error"
        exit 1
    fi

    #generate loss log
    python3 ./local/get_epoch_loss.py $logfile > $loss_log
fi

if [ $after_flavour == "plda" ]; then

    # extract xvectors
    if [ $stage -le 1 ]; then
        echo "########## extract #################" >> $logfile
        date >> $logfile
        bash ./local/run-parralel.sh extract $net_flavour $models_dir $xvectors_dir $DEVICE_COUNT $DEVICE_OFFSET >> $logfile 2>&1
    fi

    # train plda and generate target
    if [ $stage -le 2 ]; then
        echo "############ train-plda.sh ##############" >> $logfile
        date >> $logfile
        bash ./local/train-plda.sh $xvectors_dir >> $logfile 2>&1
    fi
    
    if [ $stage -le 3 ]; then
        echo "############# plda-score.sh #############" >> $logfile
        date >> $logfile
        bash ./local/plda-score.sh $xvectors_dir $eer_file >> $logfile 2>&1
    fi
else
    if [ $stage -le 2 ]; then
        echo "########### validate ###############" >> $logfile
        date >> $logfile
        bash ./local/run-parralel.sh validate $net_flavour $models_dir $scores_dir $DEVICE_COUNT $DEVICE_OFFSET >> $logfile 2>&1
    fi

    if [ $stage -le 3 ]; then
        echo "########## comput_eer ################" >> $logfile
        date >> $logfile
        bash ./local/eer.sh $eer_file $scores_dir >> $logfile 2>&1
    fi
fi

date >> $logfile
