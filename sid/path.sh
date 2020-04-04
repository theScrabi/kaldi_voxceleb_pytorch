

# Path to kaldi
KALDI=<root of kaldi project>
VOXCELEB_V2_EG=$KALDI/egs/voxceleb/v2 # Change this if you copied your VoxCeleb EG directory to somewhere else

######## TRAINING files/information here

# use this if you want to train with voxceleb v1 without silence
PRE_PROCESSED_DATA=voxceleb1_train_no_sil
# use this if you want the combination of voxceleb v1 and v2, augmented with musan and reverbation
#PRE_PROCESSED_DATA=train_combined_no_sil

export TRAIN_SCP=$VOXCELEB_V2_EG/data/$PRE_RPOCESSED_DATA/feats.scp
export SPK2UTT=$VOXCELEB_V2_EG/data/$PRE_RPOCESSED_DATA/spk2utt

export TRAINING_DEVICE=cuda:0


######### TEST/VALIDATION files here

export TRIALS_FILE=$VOXCELEB_V2_EG/data/voxceleb1_test_no_sil/trials
export TEST_SCP=$VOXCELEB_V2_EG/data/voxceleb1_test_no_sil/feats.scp
export UTT2SPK=$VOXCELEB_V2_EG/data/$PRE_RPOCESSED_DATA/utt2spk

# the first GPU used for xvector extraction
export DEVICE_OFFSET=0
# amount of GPUs used for xvector extraction.
export DEVICE_COUNT=1
