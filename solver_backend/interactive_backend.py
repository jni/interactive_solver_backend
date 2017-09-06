from __future__ import division, print_function

import itertools

import logging

import nifty

import numpy as np

import sklearn.ensemble

import struct

import threading

import zmq

from .actions import *
from .solver_utils import preprocess_with_random_forest, preprocess_with_simple_statistics
from .solver_utils import solve_multicut, node_result_to_edge_result
from .utils import cartesian_product

# logging.basicConfig( level=logging.DEBUG )

class ActionHandler( object ):

	def __init__( self ):
	    self.logger = logging.getLogger( __name__ )

	def get_solution( self, graph, costs, actions, previous_solution ):
		graph_changed = False
		for action in actions:
			g, c = self.handle_action( graph, costs, action )
			if g is not graph or c is not costs:
				graph, costs, graph_changed = g, c, True
		if graph_changed:
			self.logger.debug( 'Updated costs, resolving!' )
			solution = solve_multicut( graph, costs )
		else:
			self.logger.debug( 'No changes, using previous solution!' )
			solution = previous_solution
		return solution, graph, costs

	def handle_action( self, graph, costs, action ):
		return graph, costs

class SetCostsOnAction( ActionHandler ):

	def __init__( self, repulsive_cost=-100, attractive_cost=+100 ):
		super( SetCostsOnAction, self ).__init__()
		self.repulsive_cost  = repulsive_cost
		self.attractive_cost = attractive_cost

	def handle_action( self, graph, costs, action ):
		if isinstance( action, Detach ):
			return self._detach( graph, costs, action.fragment_id, *action.detach_from )
		elif isinstance( action, Merge ):
			return self._merge( graph, costs, *action.ids )
		else:
			return graph, costs
	
	def _merge( self, graph, costs, *ids ):
		# https://stackoverflow.com/a/942551/1725687
		if len( ids ) < 2:
			return graph, costs
		node_pairs = np.array( list( itertools.combinations( ids, 2 ) ) )

		if len( node_pairs ) == 0:
			return graph, costs
		
		edge_ids    = graph.findEdges( node_pairs )
		valid_edges = edge_ids != -1

		if not np.any( valid_edges ):
			return graph, costs
		
		edge_ids          = edge_ids[ valid_edges ]
		costs             = np.copy( costs )
		costs[ edge_ids ] = self.attractive_cost
		return graph, costs

	def _detach( self, graph, costs, fragment_id, *detach_from ):
		if len( detach_from ) == 0:
			relevant_edges = [ edge for (node, edge) in graph.nodeAdjacency( fragment_id ) ]
			if len( relevant_edges ) > 0:
				costs = np.copy( costs )
				costs[ relevant_edges ] = self.repulsive_cost
				return graph, costs
		else:
			node_pairs = [ [fragment_id, d ] for d in detach_from ]

			edge_ids    = graph.findEdges( np.array( node_pairs ) )
			valid_edges = edge_ids != -1

			if np.any( valid_edges ):
				edge_ids          = edge_ids[ valid_edges ]
				costs             = np.copy( costs )
				costs[ edge_ids ] = self.repulsive_cost
				return graph, costs
			
		return graph, costs

class TrainRandomForestFromAction( ActionHandler ):

	merge_label    = 0
	separate_label = 1

	def __init__( self, edge_features, edge_labels = None, rf = None ):
		super( TrainRandomForestFromAction, self ).__init__()
		self.logger = logging.getLogger( '{}.{}'.format( self.__module__, type( self ).__name__ ) )
		self.logger.debug( 'Insantiating {}'.format( type( self ).__name__ ) )
		self.rf            = sklearn.ensemble.RandomForestClassifier() if rf is None else rf
		self.edge_features = edge_features
		self.edge_labels   = {} if edge_labels is None else edge_labels
		self.trained_model = False

	def get_solution( self, graph, costs, actions, solution ):
		self.logger.debug( 'Getting solution' )
		graph_changed = False
		edge_labels   = self.edge_labels.copy()

		if len( actions ) == 0:
			return graph, costs, actions
		
		for action in actions:
			self.handle_action( graph, costs, action )

		self.logger.debug( 'Did handle actions -- did edge labels change? %s -- %s', edge_labels, self.edge_labels )

		current_edge_labels = np.unique( list( self.edge_labels.values() ) )
		self.logger.debug( 'Edge labels in data: %s', current_edge_labels )

		self.logger.debug( 'Already trained model? %s', self.trained_model )

		if len( self.edge_labels ) > 1 and \
			np.all(  current_edge_labels == np.array( [ 0, 1 ] ) ) and \
			( self.edge_labels != edge_labels or not self.trained_model ):
			self.logger.debug( 'Training classifier, re-solving!' )
			features = []
			labels   = []
			for k, v in self.edge_labels.items():
				features.append( self.edge_features[ k, ... ] )
				labels.append( v )
			features = np.array( features )
			labels = np.array( labels )
			self.logger.debug( 'Training classifier on features and labels: %s (%s), %s', features.shape, self.edge_features.shape, labels.shape )
			self.rf.fit( features, labels )
			self.trained_model = True
		if self.trained_model:
			probabilities = self.rf.predict_proba( self.edge_features )[ :, 1 ]
			self.logger.debug( 'Predicted probabilities: %s (edge features) %s (probabilities) %d (number of edges) %s (probabilities)', self.edge_features.shape, probabilities.shape, graph.numberOfEdges, probabilities )
			p_min = 0.001
			p_max = 1. - p_min
			probabilities = ( p_max - p_min ) * probabilities + p_min
			costs = np.log( ( 1.0 - probabilities ) / probabilities )
			solution = solve_multicut( graph, costs )
		return solution, graph, costs

	def handle_action( self, graph, costs, action ):
		self.logger.debug( 'Handling action %s', action )
		if isinstance( action, Detach ):
			return self._detach( graph, costs, action.fragment_id, *action.detach_from )
		elif isinstance( action, Merge ):
			return self._merge( graph, costs, *action.ids )
		elif isinstance( action, MergeAndDetach ):
			return self._confirm_grouping( graph, costs, action.merge_ids, action.detach_from )
		else:
			return graph, costs
	
	def _merge( self, graph, costs, *ids ):
		self.logger.debug( 'Merging ids: %s', ids )
		# https://stackoverflow.com/a/942551/1725687
		if len( ids ) < 2:
			return graph, costs
		node_pairs = np.array( list( itertools.combinations( ids, 2 ) ) )

		if len( node_pairs ) == 0:
			return graph, costs
		
		edge_ids    = graph.findEdges( node_pairs )
		valid_edges = edge_ids != -1

		if not np.any( valid_edges ):
			return graph, costs
		
		locations = edge_ids[ valid_edges ]
		self.logger.debug( 'Merges edges with ids: %s', locations )
		for edge_id in locations:
			self.edge_labels[ edge_id ] = TrainRandomForestFromAction.merge_label
		
		return graph, costs

	def _detach( self, graph, costs, fragment_id, *detach_from ):
		self.logger.debug( 'Detaching: %d from %s', fragment_id, detach_from )
		if len( detach_from ) == 0:
			relevant_edges = [ edge for (node, edge) in graph.nodeAdjacency( fragment_id ) ]
			for edge in relevant_edges:
				self.edge_labels[ edge ] = TrainRandomForestFromAction.separate_label
		else:
			node_pairs = [ [fragment_id, d ] for d in detach_from ]

			edge_ids    = graph.findEdges( np.array( node_pairs ) )
			valid_edges = edge_ids != -1

			if np.any( valid_edges ):
				locations = edge_ids[ valid_edges ]
				self.logger.debug( 'Detaching edges with ids: %s', locations )
				for edge_id in locations:
					self.edge_labels[ edge_id ] = TrainRandomForestFromAction.separate_label
			
		return graph, costs

	def _confirm_grouping( self, graph, costs, group_fragments, not_part_of_group_fragments ):
		self.logger.debug( 'Confirming: %s (merge) %s (detach)', group_fragments, not_part_of_group_fragments )
		if len( not_part_of_group_fragments ) > 0:
			for fragment_id in group_fragments:
				self._detach( graph, costs, fragment_id, *not_part_of_group_fragments )

		self._merge( graph, costs, group_fragments )

class SolverServer( object ):

	def __init__( self, graph, costs, address, initial_solution, action_handler=SetCostsOnAction() ):

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

	    # print( "Creating initial solution!" )
	    self.logger.debug( 'Creating initial solution!' )
	    self.initial_solution = initial_solution( self.graph, self.costs )
	    self.current_solution = self.initial_solution
	    self.logger.debug( 'Created initial solution: {} {}'.format( self.initial_solution, np.unique( self.initial_solution ).shape ) )
	    # print( "Created initial solution!" )
	    
	    self.action_handler = action_handler

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

		if len( ids ) < 2:
			return

		# old implementation
		# https://stackoverflow.com/a/942551/1725687
		#node_pairs = np.array( list( itertools.combinations( ids, 2 ) ) )
		#set_costs_from_uv_ids( self.graph, self.costs, node_pairs.astype(np.int32), self.attractive_cost )
		# new faster implementation
		set_costs_from_node_list( self.graph, self.costs, node_pairs.astype(np.int32), self.attractive_cost )

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
					solution, self.graph, self.costs = self.action_handler.get_solution( self.graph, self.costs, actions, self.current_solution )
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


def set_costs_from_node_list(graph, costs, node_list, value):
    edge_ids = graph.edgesFromNodeList(node_list)
    costs[edge_ids] = value


def set_costs_from_cluster_ids(graph, costs, node_labeling, cluster_u, cluster_v, value):
    # find the node-ids in the original graph belonging to the cluster-ids
    node_ids_u = np.where(node_labeling == cluster_u)[0]
    node_ids_v = np.where(node_labeling == cluster_v)[0]
    # build all possible pairs and set values (TODO is there a cheaper way to do this)
    potential_uvs = cartesian_product(node_ids_u, node_ids_v)
    set_costs_from_uv_ids(graph, costs, potential_uvs, value)


def start_server( graph, costs, address, ioThreads=1, timeout=10, initial_solution=lambda graph, costs : solve_multicut( graph, costs ), action_handler=SetCostsOnAction() ):
    server = SolverServer(
        graph=graph,
        costs=costs,
        address=address,
        initial_solution=initial_solution,
        action_handler=action_handler )
    server.start( ioThreads=ioThreads, timeout=timeout)
    return server
