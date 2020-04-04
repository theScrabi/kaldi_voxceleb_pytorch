
import numpy as np
from math import isnan
from math import floor
from time import time
from random import shuffle
from random import randint
import torch
import kaldi_io as kio
from threading import Thread
from queue import Queue
from time import sleep
from torch import nn
import traceback
import sys

def read_flavour(argv):
    if len(argv) < 2 or not (argv[1] == "stat" or argv[1] == "attent" or argv[1] == "asoft"):
        print("You need to enter a flavour as parameter: stat or attent")
        exit(1)
    return argv[1]

def save_tensor(tensor, var_name):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    np.save(var_name + ".npy", tensor)


def get_labels_and_count(spklbl_file_name):
    labels = {}
    with open(spklbl_file_name) as f:
        i = 0
        for l in f.readlines():
            labels[l.replace('\n', '')] = i
            i = i + 1
    return labels, i


def get_speaker_id(key):
    return str.split(key, "-")[0]


def get_speaker_count(spklbl_file = "./exp/spklbl"):
    return len(open(spklbl_file).readlines())


def shuffle_scp(feature_files):
    data = []
    for file_name in feature_files:
        with open(file_name) as file:
            data.extend(file.readlines())
    shuffle(data)
    outfile_name = '.tmp/' + str(time()) + '.scp'
    with open(outfile_name, 'w') as out_file:
        for l in data:
            out_file.write(l)
    return outfile_name

def get_random_frame(features, frame_size):
    # first check if the frame has exactly 400 mfccs
    # in order to prevent random from having a range of 0
    # which will result in an error
    if features.shape[0] == frame_size: 
        return features
    frame_num = randint(0, features.shape[0] - frame_size - 1)
    return features[frame_num:frame_num+frame_size]


class RandomReader:
    def __init__(self, scp_file_name):
        self.cached_scp = {}
        for line in open(scp_file_name, 'r').readlines():
            l = line.split(" ")
            key = l[0]
            ark_file_name = l[1].split(":")[0]
            seek = l[1].split(":")[1].replace("\n", "") 
            self.cached_scp[key] = (ark_file_name, int(seek))

        with open(scp_file_name) as scp_file:
            first_line = scp_file.readline().split(" ")
            self.current_ark_file = first_line[1].split(":")[0]
            self.ark_file = open(self.current_ark_file, 'rb')

    def get_mat_by_key(self, key):
        seek_pos = self.cached_scp[key][1]
        ark_file_name = self.cached_scp[key][0]
        if ark_file_name != self.current_ark_file:
            self.ark_file.close()
            self.current_ark_file = ark_file_name
            self.ark_file = open(self.current_ark_file, 'rb')
        self.ark_file.seek(seek_pos)
        return kio.read_mat(self.ark_file) 
