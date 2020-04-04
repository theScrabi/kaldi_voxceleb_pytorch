#!/usr/bin/python3

import sys

if len(sys.argv) <= 2:
    print("You need to enter the reference and hypothisis file")
    print("command <reference> <hypothisis>")
    exit(1)

ref_lbl_file = sys.argv[1]
hyp_lbl_file = sys.argv[2]

ref_lbls = []
for lbl in open(ref_lbl_file).readlines():
    lbl = lbl.replace("\n", "").split(" ")
    cid = lbl[0]
    lbl_num = lbl[1]
    speakers = lbl[2].split("-")    
    ref_lbls.append((cid, lbl_num, speakers))

hyp_lbls = []
for lbl in open(hyp_lbl_file).readlines():
    lbl = lbl.replace("\n", "").split(" ")
    cid = lbl[0]
    lbl_num = lbl[1]
    speaker = lbl[2]
    hyp_lbls.append((cid, lbl_num, speaker))

def compare(cid, lbls):
    i = 0
    for ref_lbl, hyp_lbl in lbls:
        ref_lbls = "-".join(ref_lbl[2])
        print(f"{cid} {i} {hyp_lbl[2]} {ref_lbls}")
        i += 1
    exit(1)

current_cid = ""
cur_lbls = []
for i in range(0, len(ref_lbls)):
    ref_lbl = ref_lbls[i]
    hyp_lbl = hyp_lbls[i]
    cid = ref_lbl[0]
    if not current_cid == cid and not current_cid == "":
        compare(current_cid, cur_lbls)
        cur_lbls = []
    cur_lbls.append((ref_lbl, hyp_lbl))
    current_cid = cid
    
if not len(cur_lbls) == 0:
    compare(cid, cur_lbls)

