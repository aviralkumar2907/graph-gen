import create_graphs
import utils
import networkx as nx
import tensorflow as tf
import numpy as np
from copy import deepcopy

class Dataset(object):

    def __init__(self, file_path, is_real=True, batch_size=None):
        assert batch_size is not None
        self.graph_list = utils.load_graph_list(file_path, is_real=is_real)
        self.curr_index = 0
        self.batch_size = batch_size
        self.curr_graph_list = deepcopy(self.graph_list)

    def next(self):
        n_node_types = 2
        n_edge_types = 1
        if self.curr_index <=0 or self.curr_index >= len(self.graph_list):
            random_permutation = np.random.permutation(len(self.graph_list))
            random_permutation = random_permutation.tolist()
            self.curr_list = [self.graph_list[t] for t in random_permutation]
            self.curr_index = 0
        # Try to keep batch size a factor of overall length
        graphs_in_batch = self.curr_list[self.curr_index:
                                 self.curr_index+self.batch_size]
        self.curr_index += self.batch_size
        max_graph_size = max(np.array([len(g.nodes) for g in graphs_in_batch]))
        adj_ph = np.ones([self.batch_size, max_graph_size, max_graph_size],
                         dtype=np.float32)*-20.0
        node_feature_ph = np.random.normal(
            loc=1.0, scale=0.01,
            size=(self.batch_size, max_graph_size, n_node_types))
        edge_features_ph = np.ones([self.batch_size, max_graph_size,
                                    max_graph_size, n_edge_types+1],
                                   dtype=np.float32)
        mask_ph = np.zeros([self.batch_size, max_graph_size], dtype=np.float32)

        for idx, g in enumerate(graphs_in_batch):
            adj_mat = np.asarray(nx.to_numpy_matrix(g))
            total_ineq = np.sum((adj_mat > np.transpose(adj_mat)) +
                                        (adj_mat < np.transpose(adj_mat)))
            assert total_ineq == 0, 'Make sure adj is symmetric'
            adj_ph[idx][:len(g.nodes), :len(g.nodes)] = (1.0 - adj_mat)*-20.0
            mask_ph[idx][np.arange(len(g.nodes))] = 1.0
            print ('Num nodes: ', len(g.nodes), 'Num edges: ', np.sum(adj_mat))

        adj_ph += 10.0
        return {'node_ph': node_feature_ph,
                'adj_ph': adj_ph,
                'mask_ph': mask_ph,
                'edge_features_ph': edge_features_ph}
                

def get_data_for_pass(dataset, placeholders, permute=False,
                      n_node_types=None,
                      n_edge_types=None, fake_data=False):
    """Get the data for these new graphs"""
    batch = dataset.next()
