#!/usr/bin/python3

import sys

if len(sys.argv) <= 2:
    print("command <reference> <hypothisis>")
    print("You need to enter the reference and the hypothesis file")
    exit(1)

reference_file_name = sys.argv[1]
hyp_file_name = sys.argv[2]

references = []
hyp_lbls = []

for line in open(hyp_file_name).readlines():
    line = line.replace("\n", "").split(" ")
    hyp_lbls.append(int(line[2]))



# label by start position
label_map = {}

i = 0
for line in open(reference_file_name).readlines():
    line = line.replace("\n", "").split(" ")
    cid = line[0]
    fid = line[1]
    speakers = line[2]
    starts = list(map(int, line[3].split("-")))
    ends = list(map(int, line[4].split("-")))
    hyp_lbl = hyp_lbls[i]
    label_map[starts[0]] = (starts, ends, hyp_lbl) 
    i = i + 1

def convert_time(time):
    hour = int(time / (60*60*100))
    hr_rest = time % (60*60*100)
    minute = int(hr_rest / (60*100))
    mi_rest = time % (60*100)
    seconds = int(mi_rest / 100)
    millis = seconds % 100
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hour, minute, seconds, millis)
    

def register_segment(start, stop, spkr, seg_num):
    start = convert_time(start)
    stop = convert_time(stop)
    print(seg_num)
    print(f"{start} --> {stop}")
    print(f"Flo {spkr}")
    print("")

cur_spkr = -1
seg_start = 0
last_time = 0
i = 0
for key in label_map:
    time = key 
    spkr = label_map[key][2]

    if spkr != cur_spkr:
        if not cur_spkr == -1:
            register_segment(seg_start, time, cur_spkr, i)
        seg_start = time
        cur_spkr = spkr
        i += 1

    last_time = time

register_segment(seg_start, last_time, cur_spkr, i)
