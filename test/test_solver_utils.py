import unittest
import nifty.graph.rag as nrag
import vigra
import numpy as np

# hacky import
import sys
sys.path.append('..')
from python_backend import *


class TestSolverUtils(unittest.TestCase):

    def get_data_path(self):
        raw_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi/sampleA/raw/sampleA_raw_none.h5'
        seg_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi/sampleA/ws/sampleA_wsdt_googleV1_none.h5'
        gt_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi/sampleA/gt/sampleA_neurongt_none.h5'
        return {
            'raw': raw_path,
            'seg': seg_path,
            'gt':  gt_path
        }

    def test_features(self):
        inputs = self.get_data_path()
        seg = vigra.readHDF5(inputs['seg'], 'data')
        rag = nrag.gridRag(seg)
        features = get_edge_features(rag, inputs['raw'], 'data')

        self.assertEqual(len(features), rag.numberOfEdges)
        self.assertFalse(np.isnan(features).all())
        for col in range(features.shape[1]):
            self.assertFalse((features[:, col] == 0).all())

    def test_costs(self):
        pass

    def test_preprocess(self):
        pass

    def test_mc_solver(self):
        pass


if __name__ == '__main__':
    unittest.main()
