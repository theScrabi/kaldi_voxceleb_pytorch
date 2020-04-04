#!/usr/bin/python3

import sys
import numpy as np

last_id = -1
loss = np.array([])
for line in open(sys.argv[1]).readlines():
    line = line.replace("\n", "").split(" ")
    if line[0] == "epoch":
        loss = np.append(loss, float(line[7]))
        if last_id != int(line[2]):
            last_id = int(line[2])
            print("Average loss of epoch " + str(last_id) + " is " + str(np.mean(loss)))
            loss = np.array([])

