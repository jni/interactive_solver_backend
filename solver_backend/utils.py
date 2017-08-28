import h5py
import numpy as np


def read_hdf5(path, key):
    with h5py.File(path) as f:
        data = f[key][:]
    return data


def write_hdf5(data, path, key, compression=None):
    with h5py.File(path) as f:
        if compression is None:
            f.create_dataset(key, data=data)
        else:
            f.create_dataset(key, data=data, compression=compression)


# cartesian product for arbitrary number of arrays.
# cf. https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/11146645#11146645
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
