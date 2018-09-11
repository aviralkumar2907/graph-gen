import create_graphs
import get_data
from utils import *
import networkx as nx
import numpy as np

from args import Args
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
import os
import random


if __name__ == '__main__':
    args = Args()
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)

    graphs = create_graphs.create(args)

    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_validate = graphs[0:int(0.2*graphs_len)]

    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
        graph_validate_len /= len(graphs_validate)
        print('graph_validate_len', graph_validate_len)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
        graph_test_len /= len(graphs_test)
        print('graph_test_len', graph_test_len)

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    # Test loading the graphs:
    test_file_name = args.graph_save_path + args.fname_test + '0.dat'
    g_list = load_graph_list(test_file_name, is_real=True)
    print ('G-List', g_list)
    print ('Glist 0: ', g_list[0])

    dataset = get_data.Dataset(test_file_name, is_real=True, batch_size=10)
    output = dataset.next()
    print (output['adj_ph'][0])
