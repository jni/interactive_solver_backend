import itertools
import nifty
import numpy as np
import sys
import time
import unittest

# hacky import
sys.path.append('..')
from solver_backend import set_costs_from_uv_ids, set_costs_from_node_list


class TestCostSetters(unittest.TestCase):

    def random_graph(self):
        n_nodes = 10000
        n_edges = 10 * n_nodes
        graph = nifty.graph.UndirectedGraph(n_nodes)
        edges = np.random.choice(n_nodes, size=2 * n_edges).reshape((n_edges, 2)).astype('int32')
        # remove duplicate edges
        unique_pairs = edges[:, 0] != edges[:, 1]
        edges = edges[unique_pairs]
        graph.insertEdges(edges)
        return graph

    def test_cost_setters(self):
        graph = self.random_graph()
        costs1 = np.random.rand(graph.numberOfEdges)
        costs2 = costs1.copy()

        uvs = graph.uvIds()

        # random node list
        node_list = np.random.choice(graph.numberOfNodes, size=5000, replace=False).astype('int32')

        t_0 = time.time()
        node_pairs = np.array(list(itertools.combinations(node_list, 2)))
        set_costs_from_uv_ids(graph, costs1, node_pairs, 0)
        print("Merging with 'set_costs_from_uv_ids' took: %f s" % (time.time() - t_0,))

        t_1 = time.time()
        set_costs_from_node_list(graph, costs2,  node_list, 0)
        print("Merging with 'set_costs_from_node_list' took: %f s" % (time.time() - t_1,))

        self.assertTrue(np.allclose(costs1, costs2))


if __name__ == '__main__':
    unittest.main()
