from __future__ import division, print_function

import itertools

import logging

import nifty

import numpy as np

import struct

import threading

import zmq

from .actions import *
from .solver_utils import preprocess_with_random_forest, preprocess_with_simple_statistics
from .solver_utils import solve_multicut, node_result_to_edge_result
from .utils import cartesian_product

# logging.basicConfig( level=logging.DEBUG )

class SolverServer( object ):

	def __init__( self, graph, costs, address, initial_solution, repulsive_cost=-100, attractive_cost=+100 ):

	    super( SolverServer, self ).__init__()
	    
	    self.logger = logging.getLogger( __name__ )
	    self.logger.debug( 'Instantiating server!' )

	    self.graph          = graph
	    self.costs          = np.copy( costs )
	    self.initial_costs  = costs
	    self.address        = address
	    self.context        = None
	    self.socket         = None
	    self.server_thread  = None

	    self.repulsive_cost  = repulsive_cost
	    self.attractive_cost = attractive_cost

	    # print( "Creating initial solution!" )
	    self.logger.debug( 'Creating initial solution!' )
	    self.initial_solution = initial_solution( self.graph, self.costs )
	    self.current_solution = self.initial_solution
	    self.logger.debug( 'Created initial solution: {} {}'.format( self.initial_solution, np.unique( self.initial_solution ).shape ) )
	    # print( "Created initial solution!" )

	    self.condition_object = threading.Event()
	    self.interrupted      = True

	    self.lock = threading.RLock()

	def is_running( self ):
	    with self.lock:
	        return not self.interrupted

	def start( self, ioThreads=1, timeout=10 ):

	    
		self.logger.debug( 'Starting server!' )
		with self.lock:
			if not self.is_running():
				self.interrupted   = False
				self.context       = zmq.Context.instance( ioThreads )
				self.socket        = self.context.socket( zmq.REP )
				self.socket.bind( self.address )

				target             = lambda : self._solve( timeout )
				self.server_thread = threading.Thread( target=target )
				self.server_thread.start()
		self.logger.debug( 'Started server!' )

	def stop( self ):

	    # with self.lock:
	        if self.is_running():
	            # print("stopping server!", self.context, self.socket, self.server_thread)
	            self.interrupted = True
	            self.condition_object.set()

	            context = zmq.Context.instance( 1 )
	            socket  = context.socket( zmq.REQ )
	            socket.connect( self.address )
	            socket.send_string( '' )
	            socket.recv()
	            self.condition_object.set()
	            self.server_thread.join()
	            self.socket.close()

	            self.socket        = None
	            self.context       = None
	            self.server_thread = None

	def _merge( self, *ids ):
		# https://stackoverflow.com/a/942551/1725687
		if len( ids ) < 2:
			return
		node_pairs = np.array( list( itertools.combinations( ids, 2 ) ) )
		set_costs_from_uv_ids( self.graph, self.costs, node_pairs.astype(np.int32), self.attractive_cost )

	def _detach( self, fragment_id, *detach_from ):
		if len( detach_from ) == 0:
			relevant_edges = [ edge for (node, edge) in self.graph.nodeAdjacency( fragment_id ) ]
			self.costs[ relevant_edges ] = self.repulsive_cost
		else:
			node_pairs = [ [fragment_id, d ] for d in detach_from ]
			set_costs_from_uv_ids( self.graph, self.costs, node_pairs.astype(np.int32), self.repulsive_cost )

	def _solution_to_message( self ):
		# print('translatingsolutino to message', self.current_solution, self.current_solution.shape)
		# message = bytes( 1 * 8 * self.current_solution.shape[0] )
		# struct.pack_into( 'Q' * self.current_solution.shape[0], message, *self.current_solution )
		# for k, v in enumerate( self.current_solution ):
		# 	struct.pack_into( 'QQ', message, 2 * 8 * k, k, v )
		# return message
		# return self.current_solution.byteswap().tobytes()
		# java always big endian
		# https://stackoverflow.com/questions/981549/javas-virtual-machines-endianness
		return self.current_solution.byteswap().tobytes()

	def _solve( self, timeout ):
		while self.is_running():
			self.logger.debug( 'Waiting for request at {}'.format( self.address ) )
			request = self.socket.recv_string()
			length  = len( request )
			self.logger.debug( 'Received request of length {}: `{}`.'.format( length, request ) )
			with self.lock:

				if length > 0:
					actions = Action.from_json_array( request )
					offset  = 0
					for action in actions:
						if isinstance( action, Detach ):
							self._detach( action.fragment_id, *action.detach_from )
						elif isinstance( action, Merge ):
							self._merge( *action.ids )
					self.logger.debug( 'Updated costs, resolving!' )
					solution = solve_multicut( self.graph, self.costs )
					self.logger.debug( 'Updated solution and previous solution differ at {} places'.format( np.sum( solution != self.current_solution ) ) )
					self.current_solution = solution

				# print('sending message!')
				self.logger.debug( 'Responding with current solution!' )
				self.socket.send( self._solution_to_message() )
				# print( 'sent message!' )

			# self.condition_object.wait( timeout=timeout )

# this sets the corresponding values in costs in-place
def set_costs_from_uv_ids(graph, costs, uv_pairs, values):
    edge_ids        = graph.findEdges(uv_pairs)
    valid_edges     = edge_ids != -1
    edge_ids        = edge_ids[ valid_edges ]
    # print( 'any valid edges', valid_edges, np.any( valid_edges ) )

    costs[edge_ids] = values[ valid_edges ] if isinstance( values, (np.ndarray, list, tuple) ) else values


def set_costs_from_cluster_ids(graph, costs, node_labeling, cluster_u, cluster_v, value):
    # find the node-ids in the original graph belonging to the cluster-ids
    node_ids_u = np.where(node_labeling == cluster_u)[0]
    node_ids_v = np.where(node_labeling == cluster_v)[0]
    # build all possible pairs and set values (TODO is there a cheaper way to do this)
    potential_uvs = cartesian_product(node_ids_u, node_ids_v)
    set_costs_from_uv_ids(graph, costs, potential_uvs, value)


def start_server( graph, costs, address, ioThreads=1, timeout=10, initial_solution=lambda graph, costs : solve_multicut( graph, costs ) ):
    server = SolverServer(
        graph=graph,
        costs=costs,
        address=address,
        initial_solution=initial_solution )
    server.start( ioThreads=ioThreads, timeout=timeout)
    return server
