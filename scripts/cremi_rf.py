import sys
sys.path.append('..')
from solver_backend import learn_rf


def cremi_rf(save_path):
    raw_paths = [
        '/home/papec/Work/neurodata_hdd/cremi/'
    ]
    seg_paths = []
    gt_paths = []

    learn_rf(
        input_paths,
        len(input_paths) * ['data'],
        seg_paths,
        len(seg_paths) * ['data'],
        gt_paths,
        len(gt_paths) * ['data'],
        save_path
    )


if __name__ == '__main__':
    cremi_rf('./cremi_rf.pkl')
