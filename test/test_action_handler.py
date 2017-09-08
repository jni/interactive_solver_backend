import logging
import logging.config
import nifty
import numpy as np
import os
import sklearn
import sys
import types
import unittest
import yaml

sys.path.append('..')

import solver_backend.actions             as actions
import solver_backend.interactive_backend as interactive_backend


if os.environ.get('DEBUG') is not None:
    with open('../logger.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

edges = (
    (1, 4, -10),
    (2, 3, +10),
    (2, 6, +10),
    (3, 4, -10),
    (3, 7, -10),
    (4, 5, +10),
    (4, 8, +10),
    (5, 9, +10)
)

def to_graph(edges):
    uvIds    = [(edge[0], edge[1]) for edge in edges]
    costs    = [edge[2] for edge in edges]
    node_ids = np.unique(uvIds + [(0,0)])
    graph = nifty.graph.UndirectedGraph(np.max(node_ids) + 1)
    graph.insertEdges(uvIds)
    return graph, costs

class DummyVersioning(object):

	def parent(self, version):
	    return None

class DummyRFReadWrite(object):

    def get_rf(self, version):
        return sklearn.ensemble.RandomForestClassifier()

    def get_actions(self, version):
        return []

class DummyVersionedGraphStore(object):

    def __init__(self, graph, edge_features, edge_weights):
        super( DummyVersionedGraphStore, self ).__init__()
        self.graph         = graph
        self.edge_features = edge_features
        self.edge_weights   = edge_weights

    def get_graph(self, version):
        return self.graph

    def get_edge_weights(self,version):
        return self.edge_weights

    def get_edge_features(self, version):
        return self.edge_features

class TestRandomForest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRandomForest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))

    def test_initial_labels(self):
        self.logger.debug('Testing initial labels!')
        graph, costs = to_graph(edges)

        handler = interactive_backend.TrainRandomForestFromAction(
            versioning=DummyVersioning(),
            rf_read_write=DummyRFReadWrite(),
            versioned_graph_store=DummyVersionedGraphStore(graph, costs, costs)
           )

        self.assertEqual(len(handler.actions), 0)

        # self.assertEqual(len(handler.edge_labels), 0, 'Expected zero length edge labels but got {}.'.format(handler.edge_labels,))

    def test_confirm(self):
        self.logger.debug('Testing confirm event!')

        graph, costs = to_graph(edges)

        handler = interactive_backend.TrainRandomForestFromAction(
            versioning=DummyVersioning(),
            rf_read_write=DummyRFReadWrite(),
            versioned_graph_store=DummyVersionedGraphStore(graph, costs, costs)
           )

        # This action should have any effect
        self.logger.debug('Testing nodes with no pairwise edges.')
        merge_ids   = (1, 3)
        from_ids    = (5, 6)
        action      = actions.MergeAndDetach(merge_ids, from_ids)
        edge_labels = {}
        handler.handle_action(action, handler.graph, edge_labels)
        self.assertEqual(len(edge_labels), 0, 'Expected zero length edge labels but got {}.'.format(edge_labels,))

        handler.submit_actions([action])
        self.assertEqual(len(handler.actions), 1)
        self.assertIs(handler.actions[0], action)
        handler.actions = []

        merge_ids   = (1, 3, 4)
        from_ids    = (5, 7, 8)
        action      = actions.MergeAndDetach(merge_ids, from_ids)
        edge_labels = {}
        handler.handle_action(action, handler.graph, edge_labels)
        merge_label     = interactive_backend.TrainRandomForestFromAction.merge_label
        separate_label  = interactive_backend.TrainRandomForestFromAction.separate_label
        merge_labels    = tuple(zip((0, 3), (merge_label,) * 2))
        separate_labels = tuple(zip((4, 5, 6), (separate_label,) * 3))
        labels          = merge_labels + separate_labels
        self.assertNotEqual(len(edge_labels), 0)

        self.logger.debug('Testing that all separate edges are in labels')
        for edge, label in separate_labels:
            self.assertTrue(edge in edge_labels)
            self.assertEqual(edge_labels[edge], label)

        self.logger.debug('Testing that all merge edges are in labels')
        for edge, label in merge_labels:
            self.assertTrue(edge in edge_labels)
            self.assertEqual(edge_labels[edge], label)

        self.assertEqual(len(edge_labels), len(labels))

        handler.submit_actions([action])
        self.assertEqual(len(handler.actions), 1)
        self.assertIs(handler.actions[0], action)
        handler.actions = []


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.debug('Testing action handlers...')

    unittest.main()
