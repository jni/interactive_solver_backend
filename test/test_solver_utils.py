import unittest
import numpy as np
import os

try:
    import nifty
    import nifty.graph.rag as nrag
except ImportError:
    import nifty_with_cplex as nifty
    import nifty_with_cplex.graph.rag as nrag

# hacky import
import sys
sys.path.append('..')
from solver_backend import compute_edge_features, learn_rf, solve_multicut, read_hdf5
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
        print("Running test features...")
        inputs = self.get_data_path()
        seg = read_hdf5(inputs['seg'], 'data')
        rag = nrag.gridRag(seg)
        features = compute_edge_features(rag, inputs['raw'], 'data', 8)

        self.assertEqual(len(features), rag.numberOfEdges)
        self.assertFalse(np.isnan(features).any())
        for col in range(features.shape[1]):
            self.assertFalse((features[:, col] == 0).all())
        print("... done")

    def check_preprocess_output(self, rag, costs):
        self.assertEqual(costs.ndim, 1)
        self.assertEqual(rag.numberOfEdges, len(costs))
        self.assertFalse(np.isnan(costs).any())
        self.assertFalse((costs == 0).all())

    def test_preprocess_random_forest(self):
        print("Running test preprocess random forest...")
        inputs = self.get_data_path()
        rag, costs = preprocess_with_random_forest(
            inputs['raw'], 'data',
            inputs['seg'], 'data',
            inputs['rf'], 8
        )
        self.check_preprocess_output(rag, costs)
        print("... done")

    def test_preprocess_statistics(self):
        print("Running test preprocess statistics...")
        inputs = self.get_data_path()
        for stat in ('max', 'mean', 'median', 'q75', 'q90'):
            print("with stat:", stat)
            rag, costs = preprocess_with_simple_statistics(
                inputs['pmap'], 'data',
                inputs['seg'], 'data',
                8, stat
            )
            self.check_preprocess_output(rag, costs)
        print("... done")

    def test_mc_solver(self):
        print("Running test multicut...")
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
        print("... done")

if __name__ == '__main__':
    unittest.main()
