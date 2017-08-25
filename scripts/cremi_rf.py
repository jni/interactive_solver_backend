import nifty.graph.rag as nrag
import numpy as np

import sys
sys.path.append('..')
from python_backend import learn_rf, get_edge_features, get_edge_groundtuth
from python_backend import read_hdf5


def cremi_rf(save_path):
    # TODO
    raw_paths = []
    seg_paths = []
    gt_paths = []

    features = []
    labels = []
    for ii, seg_path in enumerate(seg_paths):
        raw_path = raw_paths[ii]
        gt_path = gt_paths[ii]

        seg = read_hdf5(seg_path, 'data')
        rag = nrag.gridRag(seg)

        features.append(
            get_edge_features(rag, raw_path, 'data')
        )
        labels.append(
            get_edge_groundtuth(rag, gt_path, 'data')
        )

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    learn_rf(features, labels, save_path)
