#!/usr/bin/python3

import sys
 
key_file = sys.argv[1]
data_file = sys.argv[2]

data_map = {}
for line in open(data_file):
    line = line.replace("\n", "").split(" ")
    data_map[line[0]] = line[1]

for key in open(key_file):
    key = key.replace("\n", "")
    print(key + " " + data_map[key])
