import unittest
import vigra
import numpy as np
import os

# hacky import
import sys
sys.path.append('..')
from solver_backend import set_costs_from_uv_ids, learn_rf, preprocess_with_random_forest

class TestInteractiveBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import zipfile
        from subprocess import call

        super(TestInteractiveBackend, cls).setUpClass()

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
        super(TestInteractiveBackend, cls).tearDownClass()
        rmtree('./data')

    def get_data_path(self):
        return {
            'seg': './data/seg.h5',
            'raw': './data/raw.h5',
            'pmap': './data/pmap.h5',
            'rf':  './data/rf.pkl'
        }

    def test_set_costs(self):
        inputs = self.get_data_path()
        rag, costs = preprocess_with_random_forest(
            inputs['raw'], 'data',
            inputs['seg'], 'data',
            inputs['rf'], 8
        )

        # sample some random uv-ids
        uv_ids = rag.uvIds()
        len_sample = int(.1*len(uv_ids))
        indices = np.random.permutation(len(uv_ids))[:len_sample]
        sampled_uvs = uv_ids[indices]

        # sample some random uv-pairs
        sampled_uvs = np.concatenate(
            [sampled_uvs, np.random.randint(0, rag.numberOfNodes - 1, size=(len_sample, 2))],
            axis=0
        )

        set_costs_from_uv_ids(rag, costs, sampled_uvs, 0.)

        edge_ids = rag.findEdges(sampled_uvs)
        edge_ids = edge_ids[edge_ids != -1]
        self.assertTrue((costs[edge_ids] == 0.).all())


if __name__ == '__main__':
    unittest.main()
