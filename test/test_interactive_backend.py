import collections
import ctypes
import numpy as np
import nifty
import os
import struct
import time
import unittest
import vigra
import zmq

# hacky import
import sys
sys.path.append('..')
from solver_backend import set_costs_from_uv_ids, learn_rf, preprocess_with_random_forest, interactive_backend



edges = (
	( 1, 4, -10 ),
	( 2, 3, +10 ),
	( 2, 6, +10 ),
	( 3, 4, -10 ),
	( 3, 7, -10 ),
	( 4, 5, +10 ),
	( 4, 8, +10 ),
	( 5, 9, +10 )
	)

original_labeling = [
	0, 1, 2, 2, 4, 4, 2, 7, 4, 4
	]

def to_graph( edges ):
	uvIds    = [ (edge[0], edge[1]) for edge in edges ]
	costs    = [ edge[2] for edge in edges ]
	node_ids = np.unique( uvIds + [(0,0)] )
	graph = nifty.graph.UndirectedGraph( np.max( node_ids ) + 1 )
	graph.insertEdges( uvIds )
	return graph, costs

def relabel_to_smallest_member( solution ):
    d = collections.defaultdict( list )
    for frag, seg in enumerate( solution ):
        d[ seg ].append( frag )
    minimum_id_segments = { seg : min( frags ) for (seg, frags) in d.items() }
    return np.vectorize( minimum_id_segments.get )( solution )

class TestInteractiveBackend(unittest.TestCase):


    def test_server(self):
        print("Running test server...")
        graph, costs = to_graph( edges )
        timeout    = 10
        address = "inproc://mc-solver"

        context = zmq.Context.instance( 1 )
        socket  = context.socket( zmq.REQ )
        socket.connect( address )

        print ("Starting server!" )

        server  = interactive_backend.start_server( graph, costs, address, ioThreads=1, timeout=timeout )

        print( "Started server!", server.is_running() )

        self.assertTrue( server.is_running() )

        # get solution from no action
        print( "Check initial solution!" )
        socket.send_string( '' )
        initial_solution = np.frombuffer( socket.recv(), dtype=np.uint64 )

        self.assertTrue( np.all( initial_solution == server.current_solution ) )
        self.assertTrue( np.all( relabel_to_smallest_member( initial_solution ) == original_labeling ) )

        # send merge and evaluate
        print( "Check merge!" )
        id1 = 7
        id2 = 3
        socket.send( _merge( id1, id2  ) )
        solution = np.frombuffer( socket.recv(), dtype=np.uint64 )
        self.assertTrue( np.all( relabel_to_smallest_member( solution ) == np.array( [ 0, 1, 2, 2, 4, 4, 2, 2, 4, 4 ] ) ) )

        # send detach and evaluate
        print( "Check detach!" )
        frag_id = 4
        socket.send( _detach( frag_id ) )
        solution = np.frombuffer( socket.recv(), dtype=np.uint64 )
        self.assertTrue( np.all( relabel_to_smallest_member( solution ) == np.array( [ 0, 1, 2, 2, 4, 5, 2, 2, 8, 5 ] ) ) )

        socket.close()

        # time.sleep(seconds)
        time.sleep( timeout * 2 * 1e-3 )
        server.stop()
        self.assertFalse( server.is_running() )
        print( "Stopped server!" )

def _np_arr_to_graph( arr ):
	g = nifty.graph.UndirectedGraph( np.unique( arr ).size )
	g.deserialize( arr )
	return g

def _merge( id1, id2 ):
    # buf = ctypes.create_string_buffer( 1 * 4 + 2 * 8 )
    return struct.pack( 'QQQ', 1, id1, id2 )

def _detach( frag_id ):
    return struct.pack( 'QQ', 2, frag_id )

if __name__ == '__main__':
    unittest.main()
