### kaldi

export KALDI=<root of kaldi project>
export PATH=$PWD/utils/:$KALDI/tools/openfst/bin:$KALDI/tools/sph2pipe_v2.5:$PWD:$PATH
export PATH=$PATH:$KALDI/src/ivectorbin:$KALDI/src/featbin:$KALDI/src/bin

### diarization
VOXCELEB_V2_EG=$KALDI/egs/voxceleb/v2 # Change this if you copied your VoxCeleb EG directory to somewhere else
VALIDATIOIN_DATA_SCP=$VOXCELEB_V2_EG/data/voxceleb1_test/wav.scp
MODEL=$PWD/my_sid/exp/models/stat-nonplda/the_trained_model_that_worked_best_in_sid.pt

# needs to be the same device as the one the model was trained on
DEVICE="cuda:0"


