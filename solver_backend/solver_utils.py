from __future__ import division, print_function

import nifty.graph.rag as nrag
import nifty.graph.opt.multicut as nmc

import numpy as np

from .probability_callbacks import random_forest_callback, edge_statistics_callback
from .utils                 import read_hdf5
from functools              import partial


def node_result_to_edge_result(graph, node_result):
    uv_ids = graph.uvIds()
    return node_result[uv_ids[:, 0]] != node_result[uv_ids[:, 1]]


# TODO boundary bias ?!
def compute_edge_costs(probabilities, edge_sizes=None, max_threads=-1):

    # scale the probabilities to avoid diverging costs
    # and transform to costs
    p_min = 0.001
    p_max = 1. - p_min
    probabilities = (p_max - p_min) * probabilities + p_min
    costs = np.log((1. - probabilities) / probabilities)

    # weight by edge size
    if edge_sizes is not None:
        assert edge_sizes.shape == costs.shape
        weight = edge_sizes / edge_sizes.max()
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
    edge_sizes = nrag.accumulateMeanAndLength(
        rag, np.zeros(rag.shape, dtype='float'), numberOfThreads=max_threads
    )[0][:, 1]
    costs = compute_edge_costs(probabilities, edge_sizes=edge_sizes, max_threads=max_threads)
    return rag, costs


def preprocess_with_random_forest(
    input_paths,
    input_keys,
    fragments_path,
    fragments_key,
    rf_path,
    max_threads,
    calc_filter_2d=True
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
        input_keys = [input_keys]

    callback = partial(
        random_forest_callback,
        input_paths=input_paths,
        input_keys=input_keys,
        rf_path=rf_path,
        max_threads=max_threads,
        calc_filter_2d=calc_filter_2d
    )
    return preprocess_with_callback(fragments_path, fragments_key, callback, max_threads)


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
    solver = objective.kernighanLinFactory(
        warmStartGreedy=True
    ).create(objective)
    return solver.optimize()
