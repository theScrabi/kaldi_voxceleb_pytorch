#!/usr/bin/python3

from sys import argv
from sys import stderr
from pyannote.core import Segment, Annotation, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.metrics.segmentation import SegmentationCoverage


if len(argv) < 2:
    print("The first parameter must be the path to the segments file.")
    print("The second need to be the path to the hypothesis xvector labels")
    exit(1)

segments_file = argv[1]
hyp_lbl_file = argv[2]

frame_len = 150
half_frame = frame_len/2
quad_frame = half_frame/2
frame_stride = 75

def load_segments_file(segments_file):
    segments = Annotation()
    uem = Timeline()
    conv_segments = {}
    current_mfcc_pos = 0
    current_cid = ""
    for line in open(segments_file).readlines():
        line = line.replace("\n", "").split(" ")
        cid = line[0]
        sid = line[1]
        time_start = int(line[2])
        time_stop = int(line[3])
        duration = time_stop - time_start

        if not cid == current_cid and not current_cid == "":
            conv_segments[current_cid] = (segments, uem)
            segments = Annotation()
            uem = Timeline()
            current_mfcc_pos = 0
        segments[Segment(current_mfcc_pos/100, (current_mfcc_pos + duration)/100)] = sid
        uem.add(Segment(current_mfcc_pos/100, (current_mfcc_pos + duration)/100))
        current_cid = cid
        current_mfcc_pos += duration
    if not len(segments) == 0:
        conv_segments[current_cid] = (segments, uem)
    return conv_segments


def load_hyp_file(hyp_file):
    lbls = Annotation()
    conv_lbls = {}
    current_cid = ""

    for line in open(hyp_file).readlines():
        line = line.replace("\n", "").split(" ")
        cid = line[0]
        mfcc_num = int(line[1])
        mfcc_center_pos = mfcc_num * frame_stride + quad_frame
        lbl = line[2]
        
        if not cid == current_cid and not current_cid == "":
            conv_lbls[current_cid] = lbls
            lbls = Annotation()

        #lbls.append({"mpos": mfcc_pos, "lbl":lbl})
        lbls[Segment((mfcc_center_pos-quad_frame)/100, (mfcc_center_pos + quad_frame)/100)] = lbl
        current_cid = cid
    
    if not len(lbls) == 0:
        conv_lbls[current_cid] = lbls
    return conv_lbls

ref_segments = load_segments_file(segments_file)
hyp_labels = load_hyp_file(hyp_lbl_file)

#percision = SegmentationCoverage()
metric = DiarizationErrorRate()

for cid in ref_segments:
   val = metric(ref_segments[cid][0], hyp_labels[cid], uem=ref_segments[cid][1])
   stderr.writelines(f"{cid} DER: {val}\n")

metric.report(display=True)
