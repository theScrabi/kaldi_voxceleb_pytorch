#!/usr/bin/python3

## Uses the generated json file telling how to compile the diarization dataset
### Auf was bezieht sich das To im folgender Zeile? 
## To create the dataset. This will run ffmpeg with afcat to create the dataset

## CAUTION: A segment file generate with this script is has a different scheme than the segment
## files used by kaldi. A segment file will have this format:
## <conversation_id> <speaker> <beginning_of_segment> <exclusive_end_of_segment>

import sys
import json
from os import system

if len(sys.argv) <= 3:
    print("You need to enter the .json file representing the diarization dataset")
    print("The second parameter is the directory you want to store the wave fieles in")
    print("The thrid parameter is the diari.scp file that represents the dataset")
    exit(1)

json_file = sys.argv[1]
output_dir = sys.argv[2]
diarisation_scp = sys.argv[3]

data = json.load(open(json_file))

def mux_one_conversation(conversation, index, out_scp):
    command = "sox -v 0.99"
    output_file_name = "%05d" % index + ".wav"
    key = ""

    for stream in conversation:
        skey = stream["key"]
        start = stream["start"]
        stop = stream["stop"]
        file_name = stream["file"]
        command += f" \"|sox {file_name} -p trim {start} ={stop}\"" 

    key = "conv_id%05d" % index
    command += " -b 16 " + output_dir+"/"+output_file_name
    if system(command) != 0:
        exit(1)
    out_scp.write(key + " " + output_dir+"/"+output_file_name + "\n")

with open(diarisation_scp, "w") as out_scp:
    i = 0
    for conversation in data:
        print("Coimpile: " + str(i))
        mux_one_conversation(conversation["conv"], i, out_scp)
        i += 1




