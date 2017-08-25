import unittest
import nifty.graph.rag as nrag
import vigra
import numpy as np
import os

# hacky import
import sys
sys.path.append('..')
from python_backend import get_edge_features


class TestSolverUtils(unittest.TestCase):

    # download test data
    def setUp(self):
        import zipfile
        from subprocess import call
        url = 'https://www.dropbox.com/s/l1tgzlim8h1pb7w/test_data_anisotropic.zip?dl=0'
        data_file = 'data.zip'
        # FIXME good old wget still does the job done better than any python lib I know....
        call(['wget', '-O', data_file, url])
        with zipfile.ZipFile(data_file) as f:
            f.extractall('.')
        os.remove(data_file)

    # remove test data
    def tearDown(self):
        from shutil import rmtree
        rmtree('./data')

    def get_data_path(self):
        return {
            'seg': './data/seg.h5',
            'raw': './data/raw.h5',
            'rf': ''  # TODO
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

    def test_preprocess_random_forest(self):
        pass

    def test_preprocess_statistics(self):
        pass

    def test_mc_solver(self):
        pass


if __name__ == '__main__':
    unittest.main()
