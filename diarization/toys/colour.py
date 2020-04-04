#!/usr/bin/python3

# colsours a combined xvec lbl file

import sys
from scipy.optimize import linear_sum_assignment

colour_escape = [
    "\033[41m",
    "\033[42m",
    "\033[43m",
    "\033[44m"
]


ref_to_indx = {}
indx_to_ref = [0] * 4

def colourd_ref_lbls(refs):
    s = []
    for ref in refs: 
        s.append(colour_escape[ref_to_hyp[ref]] + ref)
    return "-".join(s)

#load lines
lines = []
for line in sys.stdin.readlines():
    lines.append(line.replace("\n", "").split(" "))


#build ref index
i = 0
for line in lines:
    lbls = line[3].split("-")
    for ref in lbls:
        if not ref in ref_to_indx:
            ref_to_indx[ref] = i
            indx_to_ref[i] = ref
            i += 1            

# this finds the best matching between labels using hungarian algorithm

print(ref_to_indx)
print(indx_to_ref)
exit(1)


for line in lines:
    # match ref lbls
    lbls = set(line[3].split("-"))
    hyp = int(line[2])
    for ref in lbls:
        if not ref in ref_to_hyp and not hyp in hyp_to_ref:
            ref_to_hyp[ref] = hyp
            hyp_to_ref[hyp] = ref

print(ref_to_hyp)
print(hyp_to_ref)

for line in lines: 
    ref_lbls = colourd_ref_lbls(set(line[3].split("-")))
    print(f"{line[0]} {line[1]} {colour_escape[int(line[2])]}{line[2]} {ref_lbls}\033[49m")
