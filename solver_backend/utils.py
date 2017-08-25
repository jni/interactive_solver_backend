import h5py


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
