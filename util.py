import networkx as nx
import numpy as np
import random
from numpy.random import randint
import torch
from prdc import compute_prdc
from sklearn.model_selection import StratifiedKFold
import os
import pickle
from scipy import linalg
from bisect import bisect_left

## GNN related utils functions

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            The graph abstraction used within Graph Isomorphism Network (GIN)
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

def generate_graph_dataset(random_graph_add):
    """
    Generate graphs from synthethic model and initialize their special representation for GIN
    :param label_dict:
    :param random_graph_add: Boolean telling if we need to augment the graph dataset with classes of random (ER) graphs
    :return:
    """
    label_dict={}
    p_ER = []
    graph_list, g_list = [], []
    mapped = len(label_dict)
    label_dict[2] = mapped
    # c_sizes = [15] * 4
    for k in range(100, 200):
        for j in range(5):
            g = nx.watts_strogatz_graph(k, 4, 0.1)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 2, [0] * g.number_of_nodes()))

    p_ER.append(sum([aa.number_of_edges() for aa in graph_list]) / sum(
        [aa.number_of_nodes() ** 2 for aa in graph_list]))
    graph_list = []
    mapped = len(label_dict)
    label_dict[3] = mapped
    # c_sizes = [15] * 4
    for k in range(100, 200):
        for j in range(5):
            g = nx.barabasi_albert_graph(k, 4)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 3, [0] * g.number_of_nodes()))

    p_ER.append(sum([aa.number_of_edges() for aa in graph_list]) / sum(
        [aa.number_of_nodes() ** 2 for aa in graph_list]))
    graph_list = []
    mapped = len(label_dict)
    label_dict[4] = mapped
    # c_sizes = [15] * 4
    for k in range(100, 200):
        for j in range(5):
            c_sizes = np.random.choice(list(range(6, 10)), 2)
            g = n_community(c_sizes, p_inter=0.01)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 4, [0] * sum(c_sizes)))

    p_ER.append(sum([aa.number_of_edges() for aa in graph_list]) / sum(
        [aa.number_of_nodes() ** 2 for aa in graph_list]) + 0.1)
    graph_list = []
    mapped = len(label_dict)
    label_dict[5] = mapped
    # c_sizes = [15] * 4
    for k in range(100, 200):
        for j in range(5):
            c_sizes = np.random.choice(list(range(30, 81)), 2)
            g = n_community(c_sizes, p_inter=0.005, pcom=0.3)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 5, [0] * sum(c_sizes)))

    p_ER.append(sum([aa.number_of_edges() for aa in graph_list]) / sum(
        [aa.number_of_nodes() ** 2 for aa in graph_list]) + 0.1)
    graph_list = []
    mapped = len(label_dict)
    label_dict[6] = mapped
    # c_sizes = [15] * 4
    for k in range(100, 200):
        for i in range(5):
            g = nx.ladder_graph(k)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 6, [0] * g.number_of_nodes()))
    graph_list = []
    mapped = len(label_dict)
    label_dict[7] = mapped
    # c_sizes = [15] * 4
    for k in range(10, 20):
        for j in range(10, 20):
            g = nx.grid_2d_graph(k, j)
            adj_matrix = nx.adjacency_matrix(g)
            g = nx.Graph(adj_matrix)
            graph_list.append(g)
            g_list.append(S2VGraph(g, 7, [0] * g.number_of_nodes()))

    p_ER.append(sum([aa.number_of_edges() for aa in graph_list]) / sum(
        [aa.number_of_nodes() ** 2 for aa in graph_list]))
    if random_graph_add:
        l = 8
        for p in p_ER:
            graph_list = []
            mapped = len(label_dict)
            label_dict[l] = mapped
            graph_list = [nx.fast_gnp_random_graph(200, p, seed=ii) for ii in
                          range(500)]
            for g in graph_list:
                g_list.append(S2VGraph(g, l, [0] * g.number_of_nodes()))
            l += 1
    return g_list,label_dict

def load_synth_data(degree_as_tag, random, onehot, random_graph_add=False, degree_feature=False):
    '''
        Create the synthetic dataset for GNN training and evaluation of techniques
        random: Choose if we use random features in FD
        onehot: Choose if we init one hot degree features or not
        random_graph_add: Choose if we want to add new graphs in the dataset
    '''
    os.makedirs('saved_graphs', exist_ok=True)
    print('loading data')

    g_list, label_dict = generate_graph_dataset(random_graph_add)

    # Create labels and edge matrices
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)
        g.label = label_dict[g.label]
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    # Initialize node tags with their degree
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique node tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    # Creating node features
    if random:
        if onehot:
            for g in g_list:
                g.node_features = torch.zeros(len(g.node_tags), len(tagset))
                g.node_features[range(len(g.node_tags)), [tag2index[tag]
                                                          for tag in g.node_tags]] = 1
        else:
            for g in g_list:
                g.node_features = torch.zeros(len(g.node_tags), 0)
    else:
        if onehot:
            for g in g_list:
                g.node_features = torch.zeros(len(g.node_tags), len(tagset))
                g.node_features[range(len(g.node_tags)), [tag2index[tag]
                                                          for tag in g.node_tags]] = 1
        elif degree_feature:
            for g in g_list:
                g.node_features = torch.Tensor(g.node_tags).reshape(len(g.node_tags), 1)
        else:
            for g in g_list:
                g.node_features = torch.ones(len(g.node_tags), 1)

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict), tag2index, len(tagset)

def load_graph_asS2Vgraph(graph_list, label, random, tag2index, lentagset, onehot=False, degree_feature=False):
    """Convert the nx.Graph list into a S2vGraph list ( preparing for GIN )"""
    g_list = []
    label_dict = {}
    feat_dict = {}
    mapped = label
    label_dict[label] = mapped

    for k in range(len(graph_list)):
        adj_matrix = nx.adjacency_matrix(graph_list[k])
        g = nx.Graph(adj_matrix)
        g_list.append(S2VGraph(g, label, [0] * graph_list[k].number_of_nodes()))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    for graph in g_list:
        graph.node_tags = list(dict(graph.g.degree).values())
        for i in range(len(graph.node_tags)):
            if graph.node_tags[i] not in tag2index.keys():
                graph.node_tags[i] = take_closest(list(tag2index.keys()), graph.node_tags[i])

    if random:
        if onehot:
            for g in g_list:
                g.node_features = torch.zeros(len(g.node_tags), lentagset)
                g.node_features[range(len(g.node_tags)), [tag2index[tag]
                                                          for tag in g.node_tags]] = 1
        else:
            for g in g_list:
                g.node_features = torch.zeros(len(g.node_tags), 0)
    else:
        if onehot:
            for g in g_list:
                g.node_features = torch.zeros(len(g.node_tags), lentagset)
                g.node_features[range(len(g.node_tags)), [tag2index[tag]
                                                          for tag in g.node_tags]] = 1
        elif degree_feature:
            for g in g_list:
                g.node_features = torch.Tensor(g.node_tags).reshape(len(g.node_tags), 1)
        else:
            for g in g_list:
                g.node_features = torch.ones(len(g.node_tags), 1)

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % lentagset)

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)

def load_graph_list(fname, is_real=True):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    return graph_list

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


## Synthethic graph generators

def n_community(c_sizes, p_inter=0.01, pcom=0.7):
    graphs = [nx.gnp_random_graph(c_sizes[i], pcom, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c) for c in nx.connected_components(G)]
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    # print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    return G


## Graph Related functions

def perturb_new_rewire(graph_list, p):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p : ration of rewiring graph
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_remove_count = 0
        for (u, v) in list(G.edges()):
            if np.random.rand()<p:
                G.remove_edge(u, v)
                edge_remove_count += 1
        # randomly add the edges back
        for i in range(edge_remove_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u,v)) and (u!=v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


## Other functions

def take_closest(myList, myNumber):
    """
    Assumes a sorted list. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before