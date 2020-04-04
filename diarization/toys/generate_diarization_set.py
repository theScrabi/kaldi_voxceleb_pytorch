#!/usr/bin/python3

## Creates a json file telling how the dataset should 
## be compiled into a diarization set

import sys
from random import randrange
import json

max_samples_per_conversation = 50
min_samples_per_conversation = 20
amount_of_conversations = 400
min_frame_length = 3
max_speaker_per_conversatioin = 4

def get_utts(file_name, utt2dur):
    utts = []
    for line in open(file_name):
        utt = line.replace("\n", "").split(" ")
        utt.append(utt2dur[utt[0]])
        utts.append(utt)
    return utts


def get_utt2dur(file_name):
    utt2dur = {}
    for line in open(file_name):
        line = line.replace("\n", "").split(" ")
        utt2dur[line[0]] = line[1]
    return utt2dur


def get_random_utt(utts):
    return utts[randrange(0, len(utts))] 


def get_random_utt_with_speakers(utts, speakers):
    while True:
        utt = get_random_utt(utts)
        sid = utt[0].split("-")[0]
        if sid in speakers:
            return utt

def make_conversation(utts):
    conversation = []
    selected_speakers = []
    for i in range(0, min_samples_per_conversation + randrange(0, max_samples_per_conversation - min_samples_per_conversation)):
        if len(selected_speakers) < max_speaker_per_conversatioin:
            utt = get_random_utt(utts)
        else:
            utt = get_random_utt_with_speakers(utts, selected_speakers)
        sid = utt[0].split("-")[0]
        if not sid in selected_speakers:
            selected_speakers.append(sid)
        dur = int(float(utt[2]))
        frame_dur = dur if dur <= min_frame_length else randrange(min_frame_length, dur+1)
        frame_pos = 0 if dur - frame_dur == 0 else randrange(0, dur - frame_dur + 1)
        conversation.append({"key":utt[0], "start":frame_pos, "stop": frame_pos + frame_dur, "file": utt[1]})

    return {"conv":conversation, "speaker":selected_speakers}

def get_conversations(utts, amount):
    conversations = []
    for rutts in range(0, amount):
        conversations.append(make_conversation(utts))
    return conversations

if len(sys.argv) <= 2:
    print("You need to enter the wav.scp as well as a utt2dur file")
    exit(1)

wav_file = sys.argv[1]
utt2dur_file = sys.argv[2]

utt2dur = get_utt2dur(utt2dur_file)
utts = get_utts(wav_file, utt2dur)
conversations = get_conversations(utts, amount_of_conversations)
print(json.dumps(conversations))

