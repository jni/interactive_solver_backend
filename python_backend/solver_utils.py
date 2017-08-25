from __future__ import division, print_function

import nifty.graph.rag as nrag
import nifty.graph.optimization.multicut as nmc

import fastfilters as ff
import pickle
import numpy as np
from functools import partial

from utils import read_hdf5


def get_edge_features(rag, input_path, input_key, max_threads=-1):

    # hard coded features
    filter_list = [ff.gaussianSmoothing, ff.laplacianOfGaussian, ff.hessianOfGaussianEigenvalues]
    sigmas = [1.6, 4.2, 8.2]

    # load the data
    data = read_hdf5(input_path, input_key)

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


def preprocess_with_callback(
    fragments_path,
    fragments_key,
    probability_callback,
    max_threads
):
    """
    Preprocess the data for the interactive solver with
    callback for edge-probability calculation.
    @Parameter
    ----------
    @Returns
    --------
    """
    assert isinstance(fragments_path, str)
    assert isinstance(fragments_key, str)

    fragments = read_hdf5(fragments_path, fragments_key)
    rag = nrag.gridRag(fragments, numberOfThreads=max_threads)

    probabilities = probability_callback(rag)
    costs = get_edge_costs(rag, probabilities)
    return rag, costs


def random_forest_callback(
    rag,
    input_paths,
    input_keys,
    rf_path,
    max_threads
):
    features = []
    for ii, path in enumerate(input_paths):
        features.append(get_edge_features(rag, path, input_keys[ii]))
    features = np.concatenate(features, axis=1)

    with open(rf_path, 'rb') as f:
        rf = pickle.load(f)

    return rf.predict_proba(features)[:, 1]


def preprocess_with_random_forest(
    input_paths,
    input_keys,
    fragments_path,
    fragments_key,
    rf_path,
    max_threads
):
    """
    Preprocess the data for the interactive solver with random forest probabilities.
    @Parameter
    ----------
    @Returns
    --------
    """

    assert isinstance(input_paths, (list, tuple, str))
    assert isinstance(input_keys, (list, tuple, str))

    if isinstance(input_paths, str):
        input_paths = [input_paths]
    if isinstance(input_keys, str):
        input_paths = [input_keys]

    callback = partial(
        random_forest_callback,
        input_paths=input_paths,
        input_keys=input_keys,
        rf_path=rf_path,
        max_threads=max_threads
    )
    return preprocess_with_callback(fragments_path, fragments_key, callback, max_threads)


def edge_statistics_callback(rag, input_path, input_key, statistic, max_threads):
    stats_to_index = {'mean': 0, 'max': 8, 'median': 5, 'q75': 6, 'q90': 7}
    assert statistic in stats_to_index
    index = stats_to_index[statistic]
    data = read_hdf5(input_path, input_key)
    return nrag.accumulateEdgeStandartFeatures(
        rag, data, data.min(), data.max(), numberOfThreads=max_threads
    )[:, index]


def preprocess_with_simple_statistics(
    input_path,
    input_key,
    fragments_path,
    fragments_key,
    max_threads,
    statistic='mean'
):
    """
    Preprocess the data for the interactive solver with edge probabilites from simple accumulation.
    @Parameter
    ----------
    @Returns
    --------
    """
    assert isinstance(input_path, str)
    assert isinstance(input_key, str)

    callback = partial(
        edge_statistics_callback,
        input_path=input_path,
        input_key=input_key,
        statistic=statistic,
        max_threads=max_threads
    )
    return preprocess_with_callback(fragments_path, fragments_key, callback, max_threads)


# TODO time limit -> needs different kl impl and visitor
# we use kernighan lin for now
def solve_multicut(graph, costs):
    assert graph.numberOfEdges == len(costs)
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.multicutAndresKernighanLinFactory(
        greedyWarmstart=True
    ).create(objective)
    return solver.optimize()


def get_edge_groundtuth(rag, gt_path, gt_key):
    gt = read_hdf5(gt_path, gt_key)
    node_gt = nrag.accumulateLabels(rag, gt)
    uv_ids = rag.uvIds()
    edge_gt = node_gt[uv_ids[:, 0]] != node_gt[uv_ids[:, 1]]
    return edge_gt


def learn_rf(
    features,
    labels,
    rf_save_path,
    n_trees=250,
    max_threads=-1
):
    assert len(features) == len(labels)
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=max_threads)
    rf.fit(features, labels)
    with open(rf_save_path, 'wb') as f:
        pickle.dump(rf, f)
