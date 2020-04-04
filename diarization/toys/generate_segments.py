#!/usr/bin/python3

import kaldi_io as kio
import sys
import json

if len(sys.argv) <= 2:
    print("command <vad.scp> <diarization.json>")
    print("The vad_data.scp and the diarization_set.json needs to be given as parameter")
    exit(1)

vad_file_name = sys.argv[1]
diarization_json = sys.argv[2]

def get_conv_info(conv, key):
    utt_pos = 0
    conv_info = []
    for utt in conv["conv"]:
        utt_duration = utt["stop"]*100 - utt["start"]*100
        speaker_id = utt["key"].split("-")[0]
        conv_info.append({"sid":speaker_id, "pos":utt_pos, "dur":utt_duration })
        utt_pos += utt_duration
    return {"key":key, "utts":conv_info}


def get_id_for_pos(conv_info, pos):
    for utt in conv_info["utts"]:
        if utt["pos"] <= pos and pos < utt["pos"] + utt["dur"]:
            return utt["sid"] 

def get_bound_of_utterence_at_pos(conv_info, pos):
    for utt in conv_info["utts"]:
        if utt["pos"] <= pos and pos < utt["pos"] + utt["dur"]:
            return utt["pos"], utt["pos"] + utt["dur"]

def get_utterences_by_time_range(conv_info, start, stop):
    touching_utts = []
    for utt in conv_info["utts"]:
        if utt["pos"] < stop and utt["pos"] + utt["dur"] >= start:
            touching_utts.append(utt)
    return touching_utts

def print_segment(key, sid, start, stop):
    print(f"{key} {sid} {start} {stop}")

def register_segment(conv_info, start, stop, utt_duration):
    stop = stop if stop < utt_duration else utt_duration # cuts of tail produced by ffmpeg
    start_id = get_id_for_pos(conv_info, start)
    stop_id = get_id_for_pos(conv_info, stop-1)
    key = conv_info["key"]
 
    utts = get_utterences_by_time_range(conv_info, start, stop)
    if len(utts) == 1:
        print_segment(key, utts[0]["sid"], start, stop)
    else:
        # split framgent into utterences if multiple utterences are touched by the segment
        start_utt = utts[0]
        end_utt = utts[len(utts)-1]
        print_segment(key, start_utt["sid"], start, start_utt["pos"] + start_utt["dur"])
        for i in range(1, len(utts)-1):
            utt = utts[i]
            print_segment(key, utt["sid"], utt["pos"], utt["pos"] + utt["dur"])
        print_segment(key, end_utt["sid"], end_utt["pos"], stop)




def get_duration(conv_info):
    dur = 0
    for utt in conv_info["utts"]:
        dur += utt["dur"]
    return dur 

data_set = json.load(open(diarization_json))

for key, vec in kio.read_vec_flt_scp(vad_file_name):
    conv_index = int(key.replace("conv_id", ""))
    conv_info = get_conv_info(data_set[conv_index], key)
    utt_duration = get_duration(conv_info)
    i = 0
    segment_start = 0
    is_in_segment = False
    for val in vec:
        if val == 1.0:
            if not is_in_segment:
                segment_start = i;
            is_in_segment = True
        else:
            if is_in_segment:
                register_segment(conv_info, segment_start, i, utt_duration)
            is_in_segment = False
                        
        i += 1
    if is_in_segment:
        register_segment(conv_info, segment_start, i, utt_duration)


