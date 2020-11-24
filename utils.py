import numpy as np
from collections import defaultdict
import sklearn.preprocessing as prep
import pickle as pkl
import sys
import scipy.sparse as sp
import networkx as nx
import json


def load_data(prefix, dataset_str):
    print('loading directed data')
    feats = np.loadtxt('{}/data/{}/feature.txt'.format(prefix, dataset_str))
    labels = np.loadtxt('{}/data/{}/group.txt'.format(prefix, dataset_str))[:, 1].astype(int)
    graph = np.loadtxt('{}/data/{}/graph.txt'.format(prefix, dataset_str)).astype(int)
    with open('{}/data/{}/clustering_adj.txt'.format(prefix, dataset_str)) as f:
        clustering_adj_json = f.read()
        clustering_adj = json.loads(clustering_adj_json)

    G = nx.Graph()
    G = G.to_undirected()
    for id in range(feats.shape[0]):
        G.add_node(id)
    for item in graph:
        G.add_edge(item[0], item[1])
        G[item[0]][item[1]]['weight'] = 1.0
    id_map = {i: i for i in range(feats.shape[0])}
    # feats[feats > 0] = 1.0
    feats /= (np.max(feats, axis=1, keepdims=True) + 1e-12)

    return G, feats, id_map, labels, clustering_adj


if __name__ == "__main__":
    load_data(".", "cora")