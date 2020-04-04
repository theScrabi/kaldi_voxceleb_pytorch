#!/bin/bash

# This script will crate the symlincs required by the kaldi commands
# ATTENTION: Run this script before you execute run.sh for the first time.

source path.sh

rm -f local
ln -s $VOXCELEB_V2_EG/local local

rm -f sid
ln -s $KALDI/egs/sre08/v1/sid sid

rm -f steps
ln -s $KALDI/egs/wsj/s5/steps steps

rm -f utils
ln -s $KALDI/egs/wsj/s5/utils utils

