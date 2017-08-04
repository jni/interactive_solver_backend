from __future__ import division, print_function

import nifty
import nifty.graph.rag as nrag
import nifty.graph.optimization.multicut as nmc
import fastfilters as ff
import vigra
import cPickle as pickle
import numpy as np


def get_edge_features(rag, input_path, input_key, max_threads=-1):

    # hard coded features
    filter_list = [ff.gaussianSmoothing, ff.laplacianOfGaussian, ff.hessianOfGaussianEigenvalues]
    sigmas = [1.6, 4.2, 8.2]

    # load the data
    data = vigra.readHDF5(input_path, input_key)

    features = []
    # iterate over the filter, compute them and then accumulate feature responses over the edges
    for filtr_fu in filter_list:
        for sigma in sigmas:
            response = filtr_fu(data, sigma)

            # for multichannel feature we need to accumulate over the channels
            if response.ndim == 3:
                features.append(
                    nrag.accumulateEdgeStandartFeatures(
                        rag, response, response.min(), response.max(), numberOfThreads=max_threads
                    )
                )
            else:
                for c in range(response.shape[-1]):
                    response_c = response[..., c]
                    features.append(
                        nrag.accumulateEdgeStandartFeatures(
                            rag, response_c, response_c.min(), response_c.max(), numberOfThreads=max_threads
                        )
                    )

    features = np.concatenate(features, axis=1)
    return np.nan_to_num(features)


# TODO boundary bias ?!
def get_edge_costs(rag, probabilities, weight_by_size=True, max_threads=-1):

    # clip the probabilities to avoid diverging costs
    # and transform to costs
    costs = np.clip(probabilities, 0.001, 0.999)
    costs = np.log((1. - probabilities) / probabilities)

    # weight by edge size
    if weight_by_size:
        fake_data = np.zeros(rag.shape, dtype='uint32')
        _, edge_size = nrag.accumulateEdgeMeaAndLenght(rag, fake_data, numberOfThreads=max_threads)
        weight = edge_size / edge_size.max()
        costs = weight * costs

    return costs


def preprocess(
    input_paths,
    input_keys,
    fragments_path,
    fragments_key,
    rf_path,
    max_threads
):
    """
    Preprocess the data for the interactive solver.
    @Parameter
    ----------
    @Returns
    --------
    """

    assert isinstance(input_paths, list)
    assert isinstance(input_keys, list)
    assert isinstance(fragments_path, str)
    assert isinstance(fragments_key, str)

    fragments = vigra.readHDF5(fragments_path, fragments_key)
    rag = nrag.gridRag(fragments, numberOfThreads=max_threads)

    features = []
    for ii, path in enumerate(input_path):
        features.append(get_edge_features(rag, path, input_keys[ii]))
    features = np.concatenate(features, axis=1)

    with open(rf_path) as f:
        rf = cPickle.load(f)

    probabilities = rf.predict_proba(features)[:, 1]
    costs = get_edge_costs(rag, probabilities)
    return rag, costs


def solve_multicut(uv_ids, costs):
    pass


def edge_groundtuth(rag, gt_path, gt_key):
    gt = vigra.readHDF5(gt_path, gt_key)
    node_gt = nrag.accumulateLabels(rag, gt)
    uv_ids = rag.uvIds()
    edge_gt = node_gt[uv_ids[:, 0]] != node_gt[uv_ids[:, 1]]
    return edge_gt


def learn_rf(
    input_paths,
    input_keys,
    fragments_path,
    fragments_key,
    gt_path,
    gt_key,
    rf_save_path,
    n_trees=250,
    max_threads=-1
):
    from sklearn.ensemble import RandomForestClassifier

    fragments = vigra.readHDF5(fragments_path, fragments_key)
    rag = nrag.gridRag(fragments, numberOfThreads=max_threads)

    features = []
    for ii, path in enumerate(input_path):
        features.append(get_edge_features(rag, path, input_keys[ii]))
    features = np.concatenate(features, axis=1)
    labels = edge_groundtuth(rag, gt_path, gt_key)

    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=max_threads)
    rf.fit(features, labels)
    with open(rf_save_path) as f:
        pickle.dump(rf, f)
