#!/usr/bin/python3

from sys import argv

print("loading")

import torch
from model import XVectorModel
from torch import nn
from utils import read_flavour
from utils import RandomReader
from utils import get_random_frame
from utils import get_speaker_count
from sys import exit
from sys import stdout
import os

speech_length = 400
amount_of_speaker = get_speaker_count()
flavour = read_flavour(argv)

# validation
trials_file = os.environ['TRIALS_FILE']
test_scp = os.environ['TEST_SCP']

if len(argv) < 4:
    print("you need to enter the model you want to validate")
    print("as well as score file name.")
    print("The last parameter need to be the device on which to run validation.")
    print("./validate.py <model_name.pt> <score_file_name> <device>")
    exit(1)

model_name = argv[2]
score_file_name  = argv[3]
device = argv[4]



# this does simple validation over the x-vectors directly

def get_features_to_compare(test_scp, trials_file):
    reader = RandomReader(test_scp)
    for line in open(trials_file).readlines():
        line = line.split(" ") 
        first_utt = reader.get_mat_by_key(line[0])
        second_utt = reader.get_mat_by_key(line[1])
        # only return frames that are long enough
        if first_utt.shape[0] >= speech_length and second_utt.shape[0] >= speech_length:
            first_utt = get_random_frame(first_utt, speech_length)
            second_utt = get_random_frame(second_utt, speech_length)
            first_utt = torch.from_numpy(first_utt).to(device)
            second_utt = torch.from_numpy(second_utt).to(device)
            first_utt = first_utt.unsqueeze(0)
            second_utt = second_utt.unsqueeze(0)
            # we need to add a batch dimension
            yield (first_utt, second_utt, line[2].replace("\n", "") == "target")


def validate(test_scp, trials_file, model, score_file=stdout):
    cosim = nn.CosineSimilarity()
    correct_classified = 0
    number_of_recordings = 0 
    for (first, second, target) in get_features_to_compare(test_scp, trials_file):
        number_of_recordings = number_of_recordings + 1
        first_xvec = model(first)
        second_xvec = model(second)
        similarity = cosim(first_xvec, second_xvec).cpu().detach().numpy()[0]
        same_speaker = False
        score_file.write(str(similarity) + (" target" if target else " nontarget") + "\n" )


print("starting")
print("validate " + score_file_name)
xmodel = XVectorModel(amount_of_speaker, flavour=flavour, learning=False, device=device)
xmodel = xmodel.to(device)
xmodel.load_state_dict(torch.load(model_name))
xmodel.eval()

validate(test_scp, trials_file, xmodel, open(score_file_name, "w"))
print("done validating " + score_file_name)
