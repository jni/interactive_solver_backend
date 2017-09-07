import collections
import ctypes
import json
import logging.config
import nifty
import numpy as np
import os
import struct
import sys
import time
import unittest
import vigra
import yaml
import zmq

# hacky import
sys.path.append('..')
from solver_backend import set_costs_from_uv_ids, learn_rf, preprocess_with_random_forest, interactive_backend, actions, solver_utils

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

original_labeling = [
    0, 1, 2, 2, 4, 4, 2, 7, 4, 4
   ]

def to_graph(edges):
    uvIds    = [(edge[0], edge[1]) for edge in edges]
    costs    = [edge[2] for edge in edges]
    node_ids = np.unique(uvIds + [(0,0)])
    graph = nifty.graph.UndirectedGraph(np.max(node_ids) + 1)
    graph.insertEdges(uvIds)
    return graph, costs

def relabel_to_smallest_member(solution):
    d = collections.defaultdict(list)
    for frag, seg in enumerate(solution):
        d[seg].append(frag)
    minimum_id_segments = { seg : min(frags) for (seg, frags) in d.items() }
    return np.vectorize(minimum_id_segments.get)(solution)


class TestInteractiveBackend(unittest.TestCase):


    def test_server(self):
        print("Running test server...")
        graph, costs = to_graph(edges)
        costs        = np.array(costs)
        timeout      = 10
        address      = "inproc://mc-solver"

        context = zmq.Context.instance(1)
        socket  = context.socket(zmq.REQ)
        socket.connect(address)

        print("Starting server!")

        action_handler = interactive_backend.SetCostsOnAction(graph, costs, solver_utils.solve_multicut(graph, costs))
        server         = interactive_backend.start_server(address, action_handler, ioThreads=1, timeout=timeout)

        print("Started server!", server.is_running())

        self.assertTrue(server.is_running())

        # get solution from no action
        print("Check initial solution!")
        socket.send_string('')
        initial_solution = np.frombuffer(socket.recv(), dtype=np.uint64).byteswap()

        # print(initial_solution, server.current_solution)
        self.assertTrue(np.all(initial_solution == action_handler.solution))
        self.assertTrue(np.all(relabel_to_smallest_member(initial_solution) == original_labeling))

        # send merge and evaluate
        print("Check merge!")
        id1 = 7
        id2 = 3
        socket.send_string(json.dumps([json.loads(_merge(id1, id2).to_json())]))
        solution = np.frombuffer(socket.recv(), dtype=np.uint64).byteswap()
        self.assertTrue(np.all(relabel_to_smallest_member(solution) == np.array([0, 1, 2, 2, 4, 4, 2, 2, 4, 4])))

        # send detach and evaluate
        print("Check detach!")
        frag_id = 4
        socket.send_string(json.dumps([json.loads(_detach(frag_id).to_json())]))
        solution = np.frombuffer(socket.recv(), dtype=np.uint64).byteswap()
        self.assertTrue(np.all(relabel_to_smallest_member(solution) == np.array([0, 1, 2, 2, 4, 5, 2, 2, 8, 5])))

        socket.close()

        # time.sleep(seconds)
        time.sleep(timeout * 2 * 1e-3)
        server.stop()
        self.assertFalse(server.is_running())
        print("Stopped server!")

def _np_arr_to_graph(arr):
    g = nifty.graph.UndirectedGraph(np.unique(arr).size)
    g.deserialize(arr)
    return g

def _merge(*ids):
    return actions.Merge(*ids)

def _detach(frag_id):
    return actions.Detach(frag_id)

if __name__ == '__main__':
    unittest.main()
