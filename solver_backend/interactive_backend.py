from __future__ import division, print_function

import zmq

from .solver_utils import preprocess_with_random_forest, preprocess_with_simple_statistics
from .solver_utils import solve_multicut, node_result_to_edge_result
from .utils import cartesian_product


# this sets the corresponding values in costs in-place
def set_costs_from_uv_ids(graph, costs, uv_pairs, value):
    edge_ids = graph.findEdges(uv_pairs)
    edge_ids = edge_ids[edge_ids != -1]
    costs[edge_ids] = value


def set_costs_from_cluster_ids(graph, costs, node_labeling, cluster_u, cluster_v, value):
    # find the node-ids in the original graph belonging to the cluster-ids
    node_ids_u = np.where(node_labeling == cluster_u)[0]
    node_ids_v = np.where(node_labeling == cluster_v)[0]
    # build all possible pairs and set values (TODO is there a cheaper way to do this)
    potential_uvs = cartesian_product(node_ids_u, node_ids_v)
    set_costs_from_uv_ids(graph, costs, potential_uvs, value)


def solver_backend():
    pass
