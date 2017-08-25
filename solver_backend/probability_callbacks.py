from __future__ import division, print_function

from concurrent import futures
import numpy as np

try:
    import nifty.graph.rag as nrag
except:
    import nifty_with_cplex.graph.rag as nrag

# we try to import fastfilters, but have vigra as a fallback
try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff
    print("Couldn't import fastfilters, using vigra as fallback -> feature computation may be slow")

import pickle
import numpy as np

from .utils import read_hdf5


def compute_2d_filter(filtr_fu, data, sigma, max_threads):
    # calculate filters for the individual slices in parallel
    with futures.ThreadPoolExecutor(max_workers=max_threads) as tp:
        tasks = [tp.submit(filtr_fu, data[z], sigma) for z in range(data.shape[0])]
        # stack the results along the z axis
        response = np.concatenate([t.result()[None, :] for t in tasks], axis=0)
    return response


def compute_edge_features(rag, input_path, input_key, max_threads, calc_filter_2d=True):

    # hard coded features
    filter_list = [ff.gaussianSmoothing, ff.laplacianOfGaussian, ff.hessianOfGaussianEigenvalues]
    sigmas = [1.6, 4.2, 8.2]

    # load the data
    data = read_hdf5(input_path, input_key)

    features = []
    # iterate over the filter, compute them and then accumulate feature responses over the edges
    for filtr_fu in filter_list:
        for sigma in sigmas:
            response = compute_2d_filter(
                filtr_fu, data, sigma, max_threads
            ) if calc_filter_2d else filtr_fu(data, sigma)

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


def random_forest_callback(
    rag,
    input_paths,
    input_keys,
    rf_path,
    max_threads,
    calc_filter_2d=True
):
    features = []
    for ii, path in enumerate(input_paths):
        features.append(
            compute_edge_features(rag, path, input_keys[ii], max_threads, calc_filter_2d)
        )
    features = np.concatenate(features, axis=1)

    with open(rf_path, 'rb') as f:
        rf = pickle.load(f)

    return rf.predict_proba(features)[:, 1]


def edge_statistics_callback(rag, input_path, input_key, statistic, max_threads):
    stats_to_index = {'mean': 0, 'max': 8, 'median': 5, 'q75': 6, 'q90': 7}
    assert statistic in stats_to_index
    index = stats_to_index[statistic]
    data = read_hdf5(input_path, input_key)
    edge_probs = nrag.accumulateEdgeStandartFeatures(
        rag, data, data.min(), data.max(), numberOfThreads=max_threads
    )[:, index]
    return np.nan_to_num(edge_probs)


