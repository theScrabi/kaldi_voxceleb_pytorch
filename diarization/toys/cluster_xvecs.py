#!/usr/bin/python3

import sys
import kaldi_io as kio
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

if len(sys.argv) <= 1:
    print("You need to enter the xvector.scp file as first parameter")

xvector_scp = sys.argv[1]
clusters_per_conv = 4

def cluster_and_label_xvects(cid, xvects):
    #clustering = KMeans(n_clusters=clusters_per_conv).fit(xvects)
    clustering = AgglomerativeClustering(n_clusters=clusters_per_conv).fit(xvects)
    labels = clustering.labels_
    for i in range(0, labels.shape[0]):
        label = labels[i]
        print(f"{cid} {i} {label}")

for cid, xvects in kio.read_mat_scp(xvector_scp):
    sys.stderr.write(f"Cluster: {cid}\n")
    cluster_and_label_xvects(cid, xvects)
