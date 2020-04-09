## Speaker identification

This file will guide you through the usage of the sid implementation.
[The scripts and their meaning](#the_scripts_and_their_meaning) will
tell you what scripts there are wand what they are used for.
[Steps](#steps) tells what you need to do to get it running successfully.


### The scripts and their meaning

First of all, you will find all scripts belonging to the execution of the model in the `/src` directory. `/local` does contain helper scripts.

- **diary_extract.py** is used for extracting xvectors for the diarization task. This script is used by `run.sh` script of the diarization implementation.
- **extract.py** is used for extracting xvectors when these should be saved and further processed by e.g. PLDA.
- **loss_functions.py** contains the implementation of the two loss functions used in this work.
- **model.py** declares the model for the deep neural network.
- **train.py** is used to control and train the model.
- **utils.py** contains handy functions. (read more about each function in the script it self).
- **validate.py** This will generate x-vectors but not save them in order to directly compare different extracted x-vectors with cosine distance.
- **get_epoch_loss.py** calculates the average loss of each epoch
- **plda-score.sh** runns plda on the x-vectors and saves the outcome as score files.
- **run-parralel.sh** is a helper script for executing `validate.py` and `extract.py` in parallel on multiple GPUs.
- **train-plda.sh** will train the PLDA.

### Steps

First run the [`run.sh`](https://github.com/kaldi-asr/kaldi/blob/master/egs/voxceleb/v2/run.sh) script from the kaldi voxceleb v2 eg **fully or at least until stage 6**.
Then the `run.sh` script from this project can be executed.
`run.sh` takes three arguments.
The first parameter tells the flavour of the model. The flavour can either be 
 - `stat` for statistical polling and regular softmax loss
 - `attent` for attention and regular softmax loss, or
 - `asoft` for statistial polling and angular softmax loss.

The second parameter tells if PLDA should be used or not. It can either be
- `plda` for plda enabled, or
- `nonplda` for plda disabled.

The third parameter is a just a word you can add in the end to identify your run.
So an example can be:
`$ ./run.sh asoft nonplda first_test_run`
Runns angular softmax without plda and calls the resulting log and eer file `asfot_nonplda_first_test_run.*`

Read more in the publication of my thesis (link <here>, if not ask me in an issue).
