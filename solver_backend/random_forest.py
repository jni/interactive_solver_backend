from __future__ import division, print_function

import nifty.graph.rag as nrag
import numpy as np
import pickle

from .probability_callbacks import compute_edge_features
from .utils                 import read_hdf5
from sklearn.ensemble       import RandomForestClassifier


def compute_edge_groundtuth(rag, gt_path, gt_key):
    gt = read_hdf5(gt_path, gt_key)
    node_gt = nrag.gridRagAccumulateLabels(rag, gt)
    uv_ids = rag.uvIds()
    edge_gt = node_gt[uv_ids[:, 0]] != node_gt[uv_ids[:, 1]]
    return edge_gt


def fit_rf(
    features,
    labels,
    rf_save_path,
    n_trees,
    max_threads
):
    assert len(features) == len(labels)

    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=max_threads)
    rf.fit(features, labels)
    with open(rf_save_path, 'wb') as f:
        pickle.dump(rf, f)


def learn_rf(
    input_paths,
    input_keys,
    fragments_paths,
    fragments_keys,
    gt_paths,
    gt_keys,
    rf_save_path,
    max_threads,
    n_trees=250,
):
    assert isinstance(input_paths, (list, tuple))
    assert isinstance(input_keys, (list, tuple))
    assert isinstance(fragments_paths, (list, tuple))
    assert isinstance(fragments_paths, (list, tuple))
    assert isinstance(gt_keys, (list, tuple))
    assert isinstance(gt_keys, (list, tuple))

    # we only allow for an even number of inputs per segmentation
    assert len(fragments_paths) == len(gt_paths)
    assert len(input_paths) % len(gt_paths) == 0
    inputs_per_seg = len(input_paths) // len(gt_paths)

    features = []
    labels = []

    # iterate over the different fragmentations and extract features and labels
    for ii, fragments_path in enumerate(fragments_paths):

        seg = read_hdf5(fragments_path, fragments_keys[ii])
        rag = nrag.gridRag(seg)
        labels.append(
            compute_edge_groundtuth(rag, gt_paths[ii], gt_keys[ii])
        )

        sub_features = []
        for jj in range(inputs_per_seg):
            inp_path = input_paths[inputs_per_seg * ii + jj]
            inp_key = input_keys[inputs_per_seg * ii + jj]
            sub_features.append(
                compute_edge_features(rag, inp_path, inp_key, max_threads)
            )

        features.append(np.concatenate(sub_features, axis=1))


    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    fit_rf(features, labels, rf_save_path, n_trees, max_threads)
