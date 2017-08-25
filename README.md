# Interactive Solver Backend

Small multicut pipeline based on nifty for use with an interacive solver.

## Usage / Examples

### Preprocessing and Multicut

In the preprocessing step, the region adjacency graph and costs for the multicut must be
calculated. This functionality is provided by `preproces_with_callback`, which accepts
paths to hdf5 files holding the input data and a callback that computes the seperation probability for edges in the graph (that will be transformed to costs automatically).
For convenience, we provide two preprocessing-functions with implemented callbacks, that 
compute the probabilities with a pretrained Random Forest / based on simple edge statistics.
Here is an example for using the Random Forest based callback and solving the multicut:

```
import nifty
from solver_backend import preprocess_with_random_forest, solve_multicut

max_number_of_threads = 8
rag, costs = preprprocess_with_random_forest(
    '/path/to/pixelmap.h5', 'hdf5_key',
    '/path/to/fragments.h5', 'hdf5_key',
    '/path/to/trained_rf.pkl', max_number_of_threads
)

graph = nifty.graph.UndirectedGraph(rag.numberOfNodes)
graph.insertEdges(rag.uvIds())
mc_nodes = solve_multicut(graph, costs)
```

To compute features from multiple input maps, pass a list of paths to h5 files as first
argument and a list of h5 keys as second argument.

#### Training Random Forest

We also provide a helper function to train a Random Forest:

```
from solver_backend import learn_rf

save_path = '/path/to/rf.pkl'
max_number_of_threads = 8
n_trees = 200
learn_rf(
    'path/to/pixelmap.h5', 'hdf5_key',
    'path/to/fragments.h5', 'hdf5_key',
    '/path/to/segmentation_gt.h5', 'hdf5_key',
    save_path, max_number_of_threads, n_trees 
)
```

### Interactive

TODO

## Requirements

- sklearn (`conda install -c conda-forge scikit-learn`)
- nifty (`conda install -c cpape nifty`)
- h5py (`conda install -c conda-forge h5py`)
- fastfilters (`conda install -c ilastik-forge fastfilters`)

If fastfilters is not working, vigra can be used as fallback (`conda install -c ilastik-forge vigra`).
