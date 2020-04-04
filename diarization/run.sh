#/bin/bash

source ./path.sh

stage=0

# This nedds to be smaller than the amaount of uterences
PARALEL_JOBS=40

if [ $stage -le 0 ]; then
    # compile raw dataset
    echo "create utt2dur"
    mkdir data/raw
    bash ./toys/make_utt2dur.sh $VALIDATIOIN_DATA_SCP > data/utt2dur
    echo "generate diraization set"
    python3 toys/generate_diarization_set.py $VALIDATIOIN_DATA_SCP data/utt2dur > ./data/diarization_set.json
    echo "compile diarization set"
    python3 toys/compile_diarization_set.py data/diarization_set.json $(pwd)/data/raw data/wav.scp
fi

if [ $stage -le 1 ]; then
    ## generate mfccs and vad
    rm ./data/utt2dur
    rm ./data/segments      # important otherwise thing go bad if you run this a second time
    awk '{print$1}' ./data/wav.scp | sort -u | awk '{print $1 " " $1}' | tee ./data/spk2utt > ./data/utt2spk
    ./utils/fix_data_dir.sh data
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $PARALEL_JOBS --cmd run.pl \
        ./data log/make_mfcc $(pwd)/data/mfccs
    ./utils/fix_data_dir.sh data
    sid/compute_vad_decision.sh --nj $PARALEL_JOBS --cmd run.pl \
        ./data exp/make_vad $(pwd)/data/vad
    ./utils/fix_data_dir.sh data
    cat ./data/vad/vad_data.*.scp | sort > ./data/vad_data.scp
    echo "generate segments"
    ./toys/generate_segments.py ./data/vad_data.scp ./data/diarization_set.json > ./data/segments
    echo "generate xvector label"
    ./toys/gen_xvec_lbl.py ./data/segments 150 75 > ./data/xvec_target_lbl
fi

if [ $stage -le 2 ]; then
    #Apply CMVN and remove silence
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj $PARALEL_JOBS --cmd run.pl \
        ./data ./data/processed ./data/processed
    utils/fix_data_dir.sh data/processed 
fi

if [ $stage -le 3 ]; then
    # extract xvectors
    cp ./my_sid/exp/spklbl ./exp/
    mkdir data/xvectors
    ./my_sid/src/diari_extract.py $MODEL ./data/processed/feats.scp ./data/xvectors/xvectors_tmp.ark $DEVICE 150 75
    copy-matrix ark:./data/xvectors/xvectors_tmp.ark ark,scp:$PWD/data/xvectors/xvectors.ark,$PWD/data/xvectors/xvectors.scp
    rm ./data/xvectors/xvectors_tmp.ark
fi

if [ $stage -le 4 ]; then
    # cluster xvectors
    # and compare them
    echo "create clusters"
    ./toys/cluster_xvecs.py ./data/xvectors/xvectors.scp > ./data/xvectors/xvec_lbl
    echo "calculate result"
    ./toys/compute_der.py data/segments data/xvectors/xvec_lbl > ./data/xvectors/result 
    cat ./data/xvectors/result 
fi

