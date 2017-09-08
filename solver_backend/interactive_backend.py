from __future__ import division, print_function

import collections
import itertools
import logging
import nifty
import numpy as np
import sklearn.ensemble
import struct
import threading
import zmq

from .actions      import *
from .solver_utils import preprocess_with_random_forest, preprocess_with_simple_statistics
from .solver_utils import solve_multicut, node_result_to_edge_result
from .solver_utils import compute_edge_costs
from .utils        import cartesian_product

# logging.basicConfig(level=logging.DEBUG)

class ActionHandler(object):

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_solution(self, graph, costs, actions, previous_solution):
        return np.array([]).astype(np.uint64)

    def submit_actions(self, version, actions):
        pass

    def update_graph(self, graph, costs, edge_features, edge_weights):
        pass

class SetCostsOnAction(ActionHandler):

    def __init__(self, graph, costs, solution, repulsive_cost=-100, attractive_cost=+100):
        super(SetCostsOnAction, self).__init__()

        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))

        self.graph           = graph
        self.costs           = costs
        self.solution        = solution
        self.repulsive_cost  = repulsive_cost
        self.attractive_cost = attractive_cost
        self.needs_mc_solve  = False

    def submit_actions(self, version, actions):
        for action in actions:
            self.handle_action(version, action)

    def handle_action(self, version, action):
        if isinstance(action, Detach):
            self._detach(action.fragment_id, *action.detach_from)
        elif isinstance(action, Merge):
            self._merge(*action.ids)

    def get_solution(self):
        if self.needs_mc_solve:
            self.solution = solve_multicut(self.graph, self.costs)
            self.needs_mc_solve = False
        return self.solution


    def _merge(self, *ids):
        # https://stackoverflow.com/a/942551/1725687
        if len(ids) < 2:
            return

        node_pairs = np.array(list(itertools.combinations(ids, 2)))

        if len(node_pairs) == 0:
            return

        edge_ids    = self.graph.findEdges(node_pairs)
        valid_edges = edge_ids != -1

        if not np.any(valid_edges):
            return

        self.logger.debug("Found edge_ids: %s", edge_ids)
        self.logger.debug("Valid edge_ids: %s", valid_edges)
        edge_ids               = edge_ids[valid_edges]
        self.logger.debug("Setting edge_ids %s to %s", edge_ids, self.attractive_cost)
        self.logger.debug("Costs: %s %s", self.costs, type(self.costs))
        self.costs[edge_ids] = self.attractive_cost
        self.needs_mc_solve    = True

    def _detach(self, fragment_id, *detach_from):
        if len(detach_from) == 0:
            relevant_edges = [edge for (node, edge) in self.graph.nodeAdjacency(fragment_id)]
            if len(relevant_edges) > 0:
                self.costs[relevant_edges] = self.repulsive_cost
                self.needs_mc_solve          = True
        else:
            node_pairs = [[fragment_id, d] for d in detach_from]

            edge_ids    = self.graph.findEdges(np.array(node_pairs))
            valid_edges = edge_ids != -1

            if np.any(valid_edges):
                edge_ids               = edge_ids[valid_edges]
                self.costs[edge_ids] = self.repulsive_cost
                self.needs_mc_solve    = True


class TrainRandomForestFromAction(ActionHandler):

    merge_label    = 0
    separate_label = 1

    def __init__(
            self,
            versioning,
            rf_read_write,
            versioned_graph_store,
            version = None,
            ):
        super(TrainRandomForestFromAction, self).__init__()
        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.logger.debug('Insantiating {}'.format(type(self).__name__))
        self.rf                    = rf_read_write.get_rf(version) # sklearn.ensemble.RandomForestClassifier() if rf is None else rf
        self.trained_model         = False
        self.retrain_rf            = True
        self.solution              = None
        self.version               = version
        self.actions               = collections.defaultdict(list)
        self.versioned_graph_store = versioned_graph_store
        self.versioning            = versioning
        self.graph                 = None
        self.edge_features         = None
        self.edge_weights          = None
        self.rf_read_write         = rf_read_write

        self._update_graph()

    def _update_graph(self):
        self.graph         = self.versioned_graph_store.get_graph(self.version)
        self.edge_features = self.versioned_graph_store.get_edge_features(self.version)
        self.edge_weights  = self.versioned_graph_store.get_edge_weights(self.version)

    def _get_features_and_labels(self):
        version  = self.version
        features = []
        labels   = []
        while version is not None:
            f, l = self._get_features_and_labels_for_version(version)
            features.append( f )
            labels.append( l )
            version = self.versioning.parent(version)


        if len(labels) == 0:
            return np.empty((0,1)), np.empty((0,))

        elif len(labels) == 1:
            return np.array(features[0]), np.array(labels[0])

        else:
            features = np.append(features[0], features[1:])
            labels   = np.append(labels[0], labels[1:])
            return features, labels

    def _get_features_and_labels_for_version(self, version):
        stored_actions = self.rf_read_write.get_actions(version)
        local_actions  = self.actions[version]
        actions        = stored_actions + local_actions
        graph          = self.graph if version == self.version else self.versioned_graph_store.get_graph(self.version)
        edge_features  = self.edge_features if version == self.version else self.versioned_graph_store.get_edge_features(self.version)
        edge_labels    = {}
        for action in actions:
            self.handle_action(action, graph, edge_labels)

        features = []
        labels   = []
        for edge_id, label in edge_labels.items():
            features.append(edge_features[ edge_id, ... ])
            labels.append(label)

        return np.array(features), np.array(labels)

    def get_solution(self):
        self.logger.debug('Getting solution')
        self.logger.debug('Version:                 %s', self.version)
        self.logger.debug('Train if enough samples: %s', self.retrain_rf)


        if self.retrain_rf:
            features, labels             = self._get_features_and_labels()
            labels_for_all_classes_exist = labels.size > 0 and np.all(np.unique(labels) == np.array([0, 1]))
            classifier_training_required = self.retrain_rf and labels_for_all_classes_exist
            self.logger.debug('Got features:             %s', features)
            self.logger.debug('Got labels:               %s', labels)
            self.logger.debug('Have all required labels: %s', labels_for_all_classes_exist)
            self.logger.debug('Re-training required:     %s', classifier_training_required)

            if classifier_training_required:

                self.rf.fit(features, labels)
                self.trained_model = True
                self.retrain_rf    = False
                probabilities      = self.rf.predict_proba(self.edge_features)[:, 1]
                # rag is still argument for compute_edge_costs
                self.costs         = compute_edge_costs(probabilities, self.edge_weights)
                self.solution      = solve_multicut(self.graph, self.costs)
                self.logger.debug('Updated solution: %s', self.solution)

        return self.solution if self.solution is not None else np.arange( self.graph.numberOfNodes, dtype=np.uint64 )


    def submit_actions(self, version, actions):
        for action in actions:
            self.logger.debug("Handling action: %s", action)
            self.actions[version].append(action)
        self.retrain_rf = True

    def handle_action(self, action, graph, edge_labels):
        self.logger.debug('Handling action %s', action)

        if isinstance(action, Detach):
            return self._detach(graph, edge_labels, action.fragment_id, *action.detach_from)
        elif isinstance(action, Merge):
            return self._merge(graph, edge_labels, *action.ids)
        elif isinstance(action, MergeAndDetach):
            return self._confirm_grouping(graph, edge_labels, action.merge_ids, action.detach_from)

    def _merge(self, graph, edge_labels, *ids):
        self.logger.debug('Merging ids: %s', ids)
        # https://stackoverflow.com/a/942551/1725687
        if len(ids) < 2:
            return False
        node_pairs = np.array(list(itertools.combinations(ids, 2)))

        if len(node_pairs) == 0:
            return False

        edge_ids    = graph.findEdges(node_pairs)
        valid_edges = edge_ids != -1

        if not np.any(valid_edges):
            return False

        locations = edge_ids[valid_edges]
        self.logger.debug('Merges edges with ids: %s', locations)
        for edge_id in locations:
            edge_labels[edge_id] = TrainRandomForestFromAction.merge_label

        return True

    def _detach(self, graph, edge_labels, fragment_id, *detach_from):
        self.logger.debug('Detaching: %d from %s', fragment_id, detach_from)
        if len(detach_from) == 0:
            relevant_edges = [edge for (node, edge) in graph.nodeAdjacency(fragment_id)]
            for edge in relevant_edges:
                edge_labels[edge] = TrainRandomForestFromAction.separate_label
            return len(relevant_edges) > 0
        else:
            node_pairs = [[fragment_id, d] for d in detach_from]

            edge_ids    = graph.findEdges(np.array(node_pairs))
            valid_edges = edge_ids != -1

            if np.any(valid_edges):
                locations = edge_ids[valid_edges]
                self.logger.debug('Detaching edges with ids: %s', locations)
                for edge_id in locations:
                    edge_labels[edge_id] = TrainRandomForestFromAction.separate_label
                return len(locations) > 0

        return False

    def _confirm_grouping(self, graph, edge_labels, group_fragments, not_part_of_group_fragments):
        self.logger.debug('Confirming: %s (merge) %s (detach)', group_fragments, not_part_of_group_fragments)
        retrain_rf = False
        if len(not_part_of_group_fragments) > 0:
            for fragment_id in group_fragments:
                retrain_rf |= self._detach(graph, edge_labels, fragment_id, *not_part_of_group_fragments)

        retrain_rf |= self._merge(graph, edge_labels, *group_fragments)

        return retrain_rf

class SolverServer(object):

    def __init__(self, address, action_handler):

        super(SolverServer, self).__init__()

        self.logger = logging.getLogger('{}.{}'.format(self.__module__, type(self).__name__))
        self.logger.debug('Instantiating server!')

        self.address        = address
        self.context        = None
        self.socket         = None
        self.server_thread  = None

        self.current_solution = action_handler.get_solution()
        self.action_handler   = action_handler

        self.condition_object = threading.Event()
        self.interrupted      = True

        self.lock = threading.RLock()

    def is_running(self):
        with self.lock:
            return not self.interrupted

    def start(self, ioThreads=1, timeout=10):


        self.logger.debug('Starting server!')
        with self.lock:
            if not self.is_running():
                self.interrupted   = False
                self.context       = zmq.Context.instance(ioThreads)
                self.socket        = self.context.socket(zmq.REP)
                self.socket.bind(self.address)

                target             = lambda : self._solve(timeout)
                self.server_thread = threading.Thread(target=target)
                self.server_thread.start()
        self.logger.debug('Started server!')

    def stop(self):

        # with self.lock:
            if self.is_running():
                # print("stopping server!", self.context, self.socket, self.server_thread)
                self.interrupted = True
                self.condition_object.set()

                context = zmq.Context.instance(1)
                socket  = context.socket(zmq.REQ)
                socket.connect(self.address)
                socket.send_string('')
                socket.recv()
                self.condition_object.set()
                self.server_thread.join()
                self.socket.close()

                self.socket        = None
                self.context       = None
                self.server_thread = None

    def _solution_to_message(self):
        # print('translatingsolutino to message', self.current_solution, self.current_solution.shape)
        # message = bytes(1 * 8 * self.current_solution.shape[0])
        # struct.pack_into('Q' * self.current_solution.shape[0], message, *self.current_solution)
        # for k, v in enumerate(self.current_solution):
        #     struct.pack_into('QQ', message, 2 * 8 * k, k, v)
        # return message
        # return self.current_solution.byteswap().tobytes()
        # java always big endian
        # https://stackoverflow.com/questions/981549/javas-virtual-machines-endianness
        return self.current_solution.byteswap().tobytes()

    def _solve(self, timeout):
        while self.is_running():
            self.logger.debug('Waiting for request at {}'.format(self.address))
            request = self.socket.recv_string()
            length  = len(request)
            self.logger.debug('Received request of length {}: `{}`.'.format(length, request))
            with self.lock:

                if length > 0:
                    json_object = json.loads(request)
                    version     = json_object['version']
                    actions     = Action.from_json_array(json.dumps(json_object['actions']))
                    self.logger.debug("Handling actions: %s", actions)
                    self.action_handler.submit_actions(version, actions)
                    # solution, self.graph, self.costs = self.action_handler.get_solution(self.graph, self.costs, actions, self.current_solution)
                # print('sending message!')
                self.logger.debug('Responding with current solution!')
                self.current_solution = self.action_handler.get_solution()
                self.socket.send(self._solution_to_message())
                # print('sent message!')

            # self.condition_object.wait(timeout=timeout)

# this sets the corresponding values in costs in-place
def set_costs_from_uv_ids(graph, costs, uv_pairs, values):
    edge_ids        = graph.findEdges(uv_pairs)
    valid_edges     = edge_ids != -1
    edge_ids        = edge_ids[valid_edges]
    # print('any valid edges', valid_edges, np.any(valid_edges))

    costs[edge_ids] = values[valid_edges] if isinstance(values, (np.ndarray, list, tuple)) else values


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


def start_server(address, action_handler, ioThreads=1, timeout=10):
    server = SolverServer(
        address=address,
        action_handler=action_handler)
    server.start(ioThreads=ioThreads, timeout=timeout)
    return server
