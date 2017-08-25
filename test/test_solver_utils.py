import unittest
import nifty
import nifty.graph.rag as nrag
import vigra
import numpy as np
import os

# hacky import
import sys
sys.path.append('..')
from solver_backend import compute_edge_features, learn_rf, solve_multicut
from solver_backend import preprocess_with_random_forest, preprocess_with_simple_statistics


class TestSolverUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import zipfile
        from subprocess import call

        super(TestSolverUtils, cls).setUpClass()

        # download the test data
        url = 'https://www.dropbox.com/s/l1tgzlim8h1pb7w/test_data_anisotropic.zip?dl=0'
        data_file = 'data.zip'
        # FIXME good old wget still does the job done better than any python lib I know....
        call(['wget', '-O', data_file, url])
        with zipfile.ZipFile(data_file) as f:
            f.extractall('.')
        os.remove(data_file)

        # learn random forest
        learn_rf(
            ['./data/raw.h5'], ['data'],
            ['./data/seg.h5'], ['data'],
            ['./data/gt.h5'], ['data'],
            './data/rf.pkl', 100
        )

    # remove test data
    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree
        super(TestSolverUtils, cls).tearDownClass()
        rmtree('./data')

    def get_data_path(self):
        return {
            'seg': './data/seg.h5',
            'raw': './data/raw.h5',
            'pmap': './data/pmap.h5',
            'rf':  './data/rf.pkl'
        }

    def test_features(self):
        inputs = self.get_data_path()
        seg = vigra.readHDF5(inputs['seg'], 'data')
        rag = nrag.gridRag(seg)
        features = compute_edge_features(rag, inputs['raw'], 'data', 8)

        self.assertEqual(len(features), rag.numberOfEdges)
        self.assertFalse(np.isnan(features).any())
        for col in range(features.shape[1]):
            self.assertFalse((features[:, col] == 0).all())

    def check_preprocess_output(self, rag, costs):
        self.assertEqual(costs.ndim, 1)
        self.assertEqual(rag.numberOfEdges, len(costs))
        self.assertFalse(np.isnan(costs).any())
        self.assertFalse((costs == 0).all())

    def test_preprocess_random_forest(self):
        inputs = self.get_data_path()
        rag, costs = preprocess_with_random_forest(
            inputs['raw'], 'data',
            inputs['seg'], 'data',
            inputs['rf'], 8
        )
        self.check_preprocess_output(rag, costs)

    def test_preprocess_statistics(self):
        inputs = self.get_data_path()
        for stat in ('max', 'mean', 'median', 'q75', 'q90'):
            rag, costs = preprocess_with_simple_statistics(
                inputs['pmap'], 'data',
                inputs['seg'], 'data',
                8, stat
            )
            self.check_preprocess_output(rag, costs)

    def test_mc_solver(self):
        inputs = self.get_data_path()
        rag, costs = preprocess_with_random_forest(
            inputs['raw'], 'data',
            inputs['seg'], 'data',
            inputs['rf'], 8
        )
        graph = nifty.graph.UndirectedGraph(rag.numberOfNodes)
        graph.insertEdges(rag.uvIds())
        mc_nodes = solve_multicut(graph, costs)
        self.assertEqual(len(mc_nodes), rag.numberOfNodes)
        self.assertGreater(len(np.unique(mc_nodes)), 10)

if __name__ == '__main__':
    unittest.main()
