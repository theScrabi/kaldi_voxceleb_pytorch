#!/usr/bin/python3

from sys import argv
from utils import read_flavour

if len(argv) <= 6:
    print("This command extracts x-vectors for diariazation based on a given model.")
    print("The frame size is entered in the amount of MFCCs. Usually on MFCC")
    print("represents 10ms of time. So a frame size of 150 represents 0.15 sec.")
    print("Frame size is an integer while frame_stride can be float point.")
    print("Run this command this way:")
    print("command <model.pt> <feats.scp> <target.ark> <device> <frame_size> <frame_stride>")

flavour = "stat"
model_file = argv[1]
source_scp = argv[2]
target_ark = argv[3]
device = argv[4]
frame_size = int(argv[5])
half_frame = frame_size/2
frame_stride = float(argv[6])

print("loading x-vector extract net")

import torch
from model import XVectorModel
import kaldi_io as kio
from math import floor
from utils import get_speaker_count
import numpy as np

amount_of_speaker = get_speaker_count()

device = torch.device(device)
xmodel = XVectorModel(amount_of_speaker, flavour=flavour, device=device, learning=False).to(device)
xmodel.load_state_dict(torch.load(model_file))
xmodel.eval()

print("starting x-vector extraction")

def get_count_of_frames(mfcc_count):
    return int((mfcc_count - frame_size) / frame_stride) + 1

def get_frames_of_conv(conv, mfccs):
    num_frames = get_count_of_frames(mfccs.shape[0]) 

    frames = []
    for i in range(0, num_frames):
        frame_center = i * frame_stride + half_frame
        frame = mfccs[floor(frame_center - half_frame): floor(frame_center + half_frame)]
        frames.append(frame)
    return np.array(frames)

with kio.open_or_fd(target_ark, "wb") as out_file:
    for conv, mfccs in kio.read_mat_scp(source_scp):
        print(f"handle conv {conv}")
        xvecs = 0
        frames = get_frames_of_conv(conv, mfccs)
        for frames_batch in np.array_split(frames, 64):
            frames_batch = torch.from_numpy(frames_batch).to(device)
            if type(xvecs) == int:
                xvecs = xmodel(frames_batch).detach().cpu().numpy()
            else:
                xvecs = np.concatenate((xvecs, xmodel(frames_batch).detach().cpu().numpy()), axis=0)
        kio.write_mat(out_file, xvecs, key=conv)
        
print("done")
