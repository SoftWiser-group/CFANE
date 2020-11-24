import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import random


class CFAggregator(nn.Module):

    def __init__(self, features,
                 input_dim,
                 output_dim,
                 adj_lists,
                 clustering_adj,
                 max_cluster,
                 num_sample=None,
                 base_model=None,
                 activation=F.elu,
                 res_rate=0.9,
                 device="cpu"):
        super(CFAggregator, self).__init__()
        print("Using CFAggregator")

        self.features = features
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj_lists = adj_lists
        self.num_sample = num_sample
        self.clustering_adj = clustering_adj
        self.max_cluster = max_cluster

        # base_model is needed, or it will omit agg1 when training
        if base_model != None:
            self.base_model = base_model
        self.activation = activation
        self.device = device
        self.agg_weight_v = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.ff_weight_v = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.weight_k = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.weight_q = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.mu = nn.Linear(self.output_dim * 2, 1, bias=False)
        self.res_rate = res_rate

    def forward(self, nodes, in_drop=0.0):

        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        cluster_neighs = [self.clustering_adj[str(int(node))] for node in nodes]
        # Local pointers to functions (speed hack)
        _set = set
        unique_nodes_list = list(set.union(*to_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        """aggregation"""
        embed_agg_attr, _ = self.features(torch.LongTensor(unique_nodes_list).to(self.device))
        self_agg_attr, self_ff_attr = self.features(nodes)
        # dropout
        embed_agg_attr = F.dropout(embed_agg_attr, p=in_drop)
        self_agg_attr = F.dropout(self_agg_attr, p=in_drop)

        """highway"""
        K = self.weight_k(torch.stack((self_agg_attr, self_ff_attr), dim=1))
        Q = self.weight_q(torch.stack((self_agg_attr, self_ff_attr), dim=1))


        embed_agg_attr = self.agg_weight_v(embed_agg_attr)
        self_agg_attr = self.agg_weight_v(self_agg_attr)

        mask = torch.zeros(len(to_neighs), self.max_cluster, len(unique_nodes)).to(self.device)
        column_indices = [unique_nodes[int(n)] for neighs in cluster_neighs for layers in neighs for n in layers]
        row_indices = [i for i in range(len(cluster_neighs)) for j in range(len(cluster_neighs[i])) for _ in range(len(cluster_neighs[i][j]))]
        layer_indices = [j for i in range(len(cluster_neighs)) for j in range(len(cluster_neighs[i])) for _ in range(len(cluster_neighs[i][j]))]
        mask[row_indices, layer_indices, column_indices] = 1
        num_neigh = mask.sum(-1, keepdim=True)
        num_neigh[num_neigh == 0] = 1
        mask = mask.div(num_neigh).to(self.device)
        neigh_agg_attr = torch.matmul(mask, embed_agg_attr)
        '''combine persona'''
        coef = torch.cat((torch.unsqueeze(self_agg_attr, dim=1).repeat(1, self.max_cluster, 1), neigh_agg_attr), dim=-1)
        coef = F.softmax(self.mu(F.normalize(coef, dim=-1)), dim=1)
        neigh_agg_attr = torch.sum(neigh_agg_attr * coef, dim=1)

        self_agg_attr = (self_agg_attr + neigh_agg_attr) / 2.0
        # self_agg_attr = neigh_agg_attr

        """feed forward"""
        self_ff_attr = self.ff_weight_v(self_ff_attr)

        V = torch.stack((self_agg_attr, self_ff_attr), dim=1)
        scores = F.softmax(torch.matmul(K, Q.transpose(-2, -1)) / self.output_dim, dim=-1)

        new_self_agg_attr, new_self_ff_attr = torch.unbind(torch.matmul(scores, V), dim=1)

        # print(torch.min(highway))
        self_agg_attr = self.res_rate * self_agg_attr + (1 - self.res_rate) * new_self_agg_attr
        self_ff_attr = self.res_rate * self_ff_attr + (1 - self.res_rate) * new_self_ff_attr

        return self.activation(self_agg_attr), self.activation(self_ff_attr)


class Aggregator(nn.Module):

    def __init__(self, features,
                 input_dim,
                 output_dim,
                 adj_lists,
                 clustering_adj,
                 max_cluster,
                 num_sample=None,
                 base_model=None,
                 activation=F.elu,
                 res_rate=0.9,
                 device="cpu"):
        super(Aggregator, self).__init__()
        print("Using Aggregator")

        self.features = features
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj_lists = adj_lists
        self.num_sample = num_sample
        self.clustering_adj = clustering_adj
        self.max_cluster = max_cluster

        # base_model is needed, or it will omit agg1 when training
        if base_model != None:
            self.base_model = base_model
        self.activation = activation
        self.device = device
        self.agg_weight_v = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.ff_weight_v = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.weight_k = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.weight_q = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.mu = nn.Linear(self.output_dim * 2, 1, bias=False)
        self.res_rate = res_rate

    def forward(self, nodes, in_drop=0.0):

        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        cluster_neighs = [self.clustering_adj[str(int(node))] for node in nodes]
        # Local pointers to functions (speed hack)
        _set = set
        unique_nodes_list = list(set.union(*to_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        """aggregation"""
        embed_agg_attr, _ = self.features(torch.LongTensor(unique_nodes_list).to(self.device))
        self_agg_attr, self_ff_attr = self.features(nodes)
        # dropout
        embed_agg_attr = F.dropout(embed_agg_attr, p=in_drop)
        self_agg_attr = F.dropout(self_agg_attr, p=in_drop)

        """highway"""
        K = self.weight_k(torch.stack((self_agg_attr, self_ff_attr), dim=1))
        Q = self.weight_q(torch.stack((self_agg_attr, self_ff_attr), dim=1))


        embed_agg_attr = self.agg_weight_v(embed_agg_attr)
        self_agg_attr = self.agg_weight_v(self_agg_attr)

        mask = torch.zeros(len(to_neighs), self.max_cluster, len(unique_nodes)).to(self.device)
        column_indices = [unique_nodes[int(n)] for neighs in cluster_neighs for layers in neighs for n in layers]
        row_indices = [i for i in range(len(cluster_neighs)) for j in range(len(cluster_neighs[i])) for _ in range(len(cluster_neighs[i][j]))]
        layer_indices = [j for i in range(len(cluster_neighs)) for j in range(len(cluster_neighs[i])) for _ in range(len(cluster_neighs[i][j]))]
        mask[row_indices, layer_indices, column_indices] = 1
        num_neigh = mask.sum(-1, keepdim=True)
        num_neigh[num_neigh == 0] = 1
        mask = mask.div(num_neigh).to(self.device)
        neigh_agg_attr = torch.matmul(mask, embed_agg_attr)
        '''combine persona'''
        coef = torch.cat((torch.unsqueeze(self_agg_attr, dim=1).repeat(1, self.max_cluster, 1), neigh_agg_attr), dim=-1)
        coef = F.softmax(self.mu(F.normalize(coef, dim=-1)), dim=1)
        neigh_agg_attr = torch.sum(neigh_agg_attr * coef, dim=1)

        self_agg_attr = (self_agg_attr + neigh_agg_attr) / 2.0
        # self_agg_attr = neigh_agg_attr

        """feed forward"""
        self_ff_attr = self.ff_weight_v(self_ff_attr)

        return self.activation(self_agg_attr), self.activation(self_ff_attr)



class AutoEncoder(nn.Module):
    def __init__(self, n_input, dim1, dim2, final_dim):
        super(AutoEncoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(n_input, dim1, bias=False),
        #     nn.Tanh(),
        #     nn.Linear(dim1, dim2, bias=False),
        #     nn.Tanh(),
        #     nn.Linear(dim2, final_dim, bias=False),
        #     nn.Tanh()
        # )
        self.decoder = nn.Sequential(
            nn.Linear(final_dim, dim2, bias=False),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(dim2, dim1, bias=False),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(dim1, n_input, bias=False)
        )
        # self.encoder.apply(init_weights)
        # self.decoder.apply(init_weights)

    def forward(self, x):
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        # return encoded, decoded
        return

    # def encode(self, x):
    #     encoded = self.encoder(x)
    #     return encoded

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded

