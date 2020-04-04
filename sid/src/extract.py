#!/usr/bin/python3

from sys import argv
from utils import read_flavour

if len(argv) < 5:
    print("This extract x-vectors based on a given model. Run it this way: command <flavour> <model.pt> <source.scp> <target.ark> <device>")
    exit(1)

flavour = read_flavour(argv)
model_file = argv[2]
source_scp = argv[3]
target_ark = argv[4]
device = argv[5]

print(f'loading {flavour} {model_file} {source_scp} {target_ark} {device}')

import torch
from model import XVectorModel
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
import kaldi_io as kio
import numpy as np
from sys import exit
import os
from threading import Thread
from queue import Queue
from utils import get_random_frame
from utils import get_speaker_count
from time import sleep

device = torch.device(device)

amount_of_speaker = get_speaker_count()
xmodel = XVectorModel(amount_of_speaker, flavour=flavour, device=device, learning=False)
xmodel = xmodel.to(device)
xmodel.load_state_dict(torch.load(model_file))
xmodel.eval()

batch_queue = Queue()
write_queue = Queue()
keep_writing = False

def get_one_batch(scp):
    batch_x = []
    batch_y = []
    for key, mat in kio.read_mat_scp(scp):
        if mat.shape[0] >= 400:
            batch_x.append(get_random_frame(mat, 400))
            batch_y.append(key)
            if len(batch_x) >= 64:
                yield (batch_y, np.array(batch_x))
                batch_x = [] 
                batch_y = []
    yield (batch_y, np.array(batch_x))


def batch_loader_thread(scp_file):
    for batch in get_one_batch(scp_file):
        batch_queue.put(batch)

def batch_writer_thread(ark_file):
    with kio.open_or_fd(ark_file, 'wb') as output_file:
        while keep_writing or (not write_queue.empty()):
            while not write_queue.empty():
                (keys, vecs) = write_queue.get()
                vecs = vecs.numpy()
                for key, vec in zip(keys, vecs):
                    kio.write_vec_flt(output_file, vec, key=key)
            sleep(0.01) 

def extractXvectors(input_scp, output_ark, model):
    with kio.open_or_fd(output_ark, 'wb') as out_file:
        source_t = Thread(target=batch_loader_thread, args=(input_scp,), daemon=True)
        source_t.start()
        target_t = Thread(target=batch_writer_thread, args=(output_ark,))
        global keep_writing
        keep_writing = True
        target_t.start()
        while source_t.isAlive():
            while not batch_queue.empty():
                keys, mats = batch_queue.get()
                x = torch.from_numpy(mats).to(device)
                y = model(x).detach().cpu()
                write_queue.put((keys, y))
            sleep(0.01)
        keep_writing = False


xmodel = XVectorModel(amount_of_speaker, flavour=flavour, device=device, learning=False)
xmodel = xmodel.to(device)
xmodel.load_state_dict(torch.load(model_file))
xmodel.eval()

print(f'starting {flavour} {model_file} {source_scp} {target_ark} {device}')
extractXvectors(source_scp, target_ark, model=xmodel)

