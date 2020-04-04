#/usr/bin/python3

from sys import argv

print("loading")
import torch
import kaldi_io as kio
import numpy as np
from model import XVectorModel
from torch import nn
from torch import optim
from sys import exit
import os
from math import isnan
from math import floor
from queue import Queue
from threading import Thread
from time import sleep
from utils import shuffle_scp
from utils import get_speaker_id
from utils import read_flavour
from utils import get_labels_and_count
from utils import get_random_frame
from utils import get_speaker_count



speech_length = 400
batch_size = 64
amount_of_speaker = get_speaker_count()
learning_rate = 0.001
train_scp = os.environ['TRAIN_SCP']
spklbl_file_name = "./exp/spklbl"
device = torch.device(os.environ['TRAINING_DEVICE'])


flavour = read_flavour(argv)

if len(argv) < 3:
    print("you need to enter a directory for the models")
    exit(1)
model_dir = argv[2]

labels, amount_of_speaker = get_labels_and_count(spklbl_file_name)
xmodel = XVectorModel(amount_of_speaker, flavour=flavour, device=device, learning=True).to(device)
optimizer = optim.Adam(xmodel.parameters(), lr=learning_rate)

batch_queue = Queue()
 
# used for debugging if NaN values start to come up again
#torch.autograd.set_detect_anomaly(True)

def set_speech_length(speech_length):
    speech_length = speech_length


def id_to_vector(id):
    vector = np.zeros([amount_of_speaker])
    vector[id - 1] = 1
    return vector


def train_one_batch(batch_x, batch_y, epochnum, batchnum):
    optimizer.zero_grad()
    x = torch.from_numpy(batch_x).to(device)
    target = torch.from_numpy(batch_y).type(torch.long).to(device)
    loss, penalty = xmodel(x, target)

    (loss + penalty).backward()
    optimizer.step()

    # train
    if isnan(loss):
        print("************************** HALT AND CATCH FIRE *****************************")
        exit(1)

    print("epoch : " + str(epochnum) + " batch: " + str(batchnum) + " Loss is: " + str(float(loss)))

def get_one_batch_with_random_frames(shuffled_scp_file):
    batch_x = []
    batch_y = []
    for key, mat in kio.read_mat_scp(shuffled_scp_file):
        if mat.shape[0] >= speech_length:
            x = get_random_frame(mat, speech_length)
            # y = id_to_vector(labels[get_speaker_id(key)]) # use this for mse loss
            y = labels[get_speaker_id(key)] # use this for cross entropy loss
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) == 64:
                yield (np.array(batch_x), np.array(batch_y))
                batch_x = []
                batch_y = []


def get_one_batch_with_sequential_frames(shuffled_scp_file):
    batch_x = []
    batch_y = []
    for key, mat in kio.read_mat_scp(shuffled_scp_file):
        # y = id_to_vector(labels[get_speaker_id(key)]) # use this for mse loss
        y = labels[get_speaker_id(key)] # use this for cross entropy loss
        for i in range(0, floor(mat.shape[0] / speech_length)):
            start = i * speech_length
            stop = start + speech_length
            batch_x.append(mat[start:stop])
            batch_y.append(y)
            if len(batch_x) == 64:
                yield (np.array(batch_x), np.array(batch_y))
                batch_x = []
                batch_y = []



def batch_loader_thread(scp_files):
    shuffled_scp_file = shuffle_scp(scp_files)
    for batch in get_one_batch_with_sequential_frames(shuffled_scp_file):
        batch_queue.put(batch)
            

def train_epoch(epoch_num, scp_files):
    batch_num = 0
    t = Thread(target=batch_loader_thread, args=([scp_files]), daemon=True)
    t.start()
    while t.isAlive():
        while not batch_queue.empty():
            (batch_x, batch_y) = batch_queue.get()
            train_one_batch(batch_x, batch_y, epoch_num, batch_num)
            batch_num = batch_num + 1
        # if main thread is to fast it gets penalized
        sleep(0.01)
        
def predict_id(x):
    batch_x = torch.from_numpy(x)
    batch_x = batch_x.unsqueeze(0).to(device)
    calc_x = xmodel(batch_x)
    return calc_x.cpu().detach().numpy()[0].argmax()

def train(num_of_epochs, scp_files, model_dir=""):
    for i in range(0, num_of_epochs):
        train_epoch(i, scp_files)
        if not model_dir == "":
            torch.save(xmodel.state_dict(), model_dir + "/raw_model" + ("%02d" % i) + ".pt")

# training 

print("starting")
train(num_of_epochs=20,  model_dir=model_dir, scp_files=[train_scp])

