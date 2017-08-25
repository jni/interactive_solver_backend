from __future__ import division, print_function

import zmq
from .solver_utils import preprocess_with_random_forest, preprocess_with_simple_statistics
from .solver_utils import solve_multicut, node_result_to_edge_result


# this sets the corresponding values in costs in-place
def set_costs_from_uv_ids(graph, costs, uv_pairs, value):
    edge_ids = graph.findEdges(uv_pairs)
    edge_ids = edge_ids[edge_ids != -1]
    costs[edge_ids] = value


def solver_backend():
    pass
