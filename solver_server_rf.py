import logging
import logging.config
import nifty
import numpy as np
import os
import signal
import sys
import time
import yaml

import solver_backend

if __name__ == "__main__":
    import argparse

    default_level = 'INFO'

    parser = argparse.ArgumentParser()
    parser.add_argument( '--graph'          , '-g', default='/data/hanslovskyp/constantin-example-data/data/mc-graph.npy' )
    parser.add_argument( '--costs'          , '-c', default='/data/hanslovskyp/constantin-example-data/data/mc-costs.npy' )
    parser.add_argument( '--weights'        , '-w', default='/data/hanslovskyp/constantin-example-data/data/edge-weights.npy' )
    parser.add_argument( '--features'       , '-f', default='/data/hanslovskyp/constantin-example-data/data/edge-features.npy' )
    parser.add_argument( '--address'        , '-a', default='ipc:///tmp/mc-solver' )
    parser.add_argument( '--logging-config' , '-l', default='/home/hanslovskyp/workspace/bigcat-future/interactive_solver_backend/logger.yaml' )

    args    = parser.parse_args()
    costs   = np.load( args.costs, allow_pickle=False )
    weights = np.load( args.weights, allow_pickle=False )
    address = args.address
    graph   = nifty.graph.UndirectedGraph()

    graph.deserialize( np.load( args.graph, allow_pickle=False ) )

    try:
        with open( args.logging_config, 'r' ) as f:
            config = yaml.safe_load( f.read() )
        logging.config.dictConfig( config )
    except:
        try:
            logging.basicConfig( level=args.logging_config )
        except:
            logging.basicConfig( level=default_level )


    def initial_solution( graph, costs ):
        solution = solver_backend.solve_multicut( graph, costs )
        print (" Got solution!")
        return solution

    edge_features  = np.load( args.features, allow_pickle=False )
    costs          = np.zeros( ( graph.numberOfEdges, ), dtype=np.float64 )
    action_handler = solver_backend.TrainRandomForestFromAction( graph, costs, edge_features=edge_features, edge_weights=weights )
    server         = solver_backend.SolverServer( address, action_handler=action_handler )
    server.start()

    def handle_signal_interrupt( signal, frame ):
        print( "Stopping server!" )
        server.stop()
    signal.signal( signal.SIGINT, handle_signal_interrupt )

