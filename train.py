import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import argparse
import math
from model import MultiviewModel
from aggregators import CFAggregator, Aggregator, AutoEncoder
from utils import load_data
from sklearn.utils import shuffle as sk_shuffle
import matplotlib.pyplot as plt
from sklearn import manifold
import node2vec as node2vec


parser = argparse.ArgumentParser()


parser.add_argument('--walk', type=bool, default=True, help='Using random walk as context')
parser.add_argument('--p', type=float, default=1.0, help='node2vec p')
parser.add_argument('--q', type=float, default=1.0, help=' node2vec q')
parser.add_argument('--num_walks', type=int, default=10, help='number of walks')
parser.add_argument('--walk_length', type=int, default=40, help='walk length')
parser.add_argument('--window_size', type=int, default=5, help='window size')
parser.add_argument('--directed', type=bool, default=False, help='directed edge')

parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--test_batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--epochs', type=int, default=10000, help='epochs')
parser.add_argument('--max_iter', type=int, default=2500, help='max iteration')
parser.add_argument('-lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--fin_lr', type=float, default=0.0001, help='final learning rate')
parser.add_argument('-weight_decay', type=float, default=0.000, help='weight_decay')
parser.add_argument('--neg_samples', type=int, default=20, help='number of negative sampling')
parser.add_argument('--neg_weights', type=int, default=10, help='weight of negative samples')
parser.add_argument('-in_drop', type=float, default=0.0, help='input sequence dropout rate')
parser.add_argument('-coef_drop', type=float, default=0.0, help='coefficient dropout rate(For attention aggregator)')

parser.add_argument('--output_dim1', type=int, default=512, help='the output dim of layer1, 2x if gcn1==False')
parser.add_argument('--output_dim2', type=int, default=256, help='the output dim of layer2, 2x if gcn2==False')
parser.add_argument('--final_dim', type=int, default=128, help='the output dim of final layer')
parser.add_argument('--num_sample1', type=int, default=None, help='number of samples in layer1.(None means full field)')
parser.add_argument('--num_sample2', type=int, default=None, help='number of samples in layer2.(None means full field)')
parser.add_argument('--loss_fn', default='skipgram', help='loss function')

parser.add_argument('--print_every', type=int, default=50, help='epochs')
parser.add_argument('--view_selection', type=bool, default=True, help='Using view selection')
parser.add_argument('--cross_fusion', type=bool, default=True, help='Using Cross fusion')


parser.add_argument('--prefix', type=str, default='.', help='dir prefix')
parser.add_argument('--dataset', type=str, default='cora', help='dataset name')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print('training on', device)
print(torch.cuda.device_count(), "GPUs")


def train(train_data):

    G = train_data[0]
    feat_data = train_data[1]
    id2idx = train_data[2]
    idx2id = {idx: id for id, idx in id2idx.items()}
    adj_lists = [set(list(G.neighbors(idx2id[i]))) for i in idx2id]
    labels = train_data[3]
    # num_classes = labels.max() + 1
    # print("num_classes", num_classes)
    clustering_adj = train_data[4]
    max_cluster = 0
    for node in clustering_adj:
        max_cluster = max(max_cluster, len(clustering_adj[node]))

    num_nodes, feats_dim = feat_data.shape
    _, struct_dim = feat_data.shape
    features = nn.Embedding(num_nodes, feats_dim).to(device)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data).to(device), requires_grad=False)

    # aggregator
    if args.cross_fusion == True:
        agg = CFAggregator
    else:
        agg = Aggregator



    agg1 = agg(features=lambda nodes: (features(nodes), features(nodes)),
                      input_dim=feats_dim,
                      output_dim=args.output_dim1,
                      adj_lists=adj_lists,
                        clustering_adj=clustering_adj,
                        max_cluster=max_cluster,
                      num_sample=args.num_sample1,
                      activation=lambda x: F.leaky_relu(x, negative_slope=0.5),
                      device=device)
    agg2 = agg(features=lambda nodes: agg1(nodes),
                      input_dim=args.output_dim1,
                      output_dim=args.output_dim2,
                      adj_lists=adj_lists,
                        clustering_adj=clustering_adj,
                        max_cluster=max_cluster,
                      num_sample=args.num_sample2,
                      base_model=agg1,
                      activation=lambda x: x,
                      device=device)

    autoencoder = AutoEncoder(feats_dim,
                              args.output_dim1,
                              args.output_dim2,
                              args.final_dim if args.view_selection else 2 * args.final_dim)

    model = MultiviewModel(args.final_dim,
                               features=features,
                               last_agg=agg2,
                               autoencoder=autoencoder,
                               neg_weight=args.neg_weights,
                               loss_fn=args.loss_fn,
                           view_selection=args.view_selection,
                           cross_fusion=args.cross_fusion)
    model.to(device)

    context = list(G.edges())
    if args.walk:
        n2v_G = node2vec.Graph(G, args.directed, args.p, args.q)
        n2v_G.preprocess_transition_probs()
        walk = n2v_G.simulate_walks(args.num_walks, args.walk_length)
        context.extend(generate_context(walk, args.window_size))

    node_list = list(G.nodes())
    alist = [item[0] for item in context]
    blist = [item[1] for item in context]
    degrees = [len(list(G.neighbors(idx2id[i]))) for i in idx2id]
    degrees = np.array(degrees)
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    iter = 0
    for epoch in range(args.epochs):
        alist, blist = sk_shuffle(alist, blist)
        start = 0
        while start * args.batch_size < len(alist):
            if iter >= 2100:
                task = 0
            elif (iter) % 3 <= 1:
                task = 1
            else:
                task = 2

            if iter > 500 and iter % 100 == 0:
                lr = args.fin_lr + (args.lr - args.fin_lr) * math.exp(-iter / 1000)
                for p in optimizer.param_groups:
                    p['lr'] = lr
                print("Changing lr to {:.6f}".format(lr))
            batch_a = alist[args.batch_size * start: args.batch_size * (start+1)]
            batch_b = blist[args.batch_size * start: args.batch_size * (start+1)]
            prop = degrees.copy()
            prop[batch_a] = 0
            prop[batch_b] = 0
            prop = np.power(prop, 0.75)
            prop = prop / np.sum(prop)
            batch_neg = np.random.choice(node_list, args.neg_samples, p=prop)
            optimizer.zero_grad()
            loss = model.loss(torch.LongTensor(batch_a).to(device),
                              torch.LongTensor(batch_b).to(device),
                              torch.LongTensor(batch_neg).to(device),
                              in_drop=args.in_drop,
                              task=task)
            loss.backward()
            torch.nn.utils.clip_grad_value_(filter(lambda p: p.requires_grad, model.parameters()), 5)
            optimizer.step()
            if iter % args.print_every == 0:
                print("Iter {}\tloss: {:.5f}".format(iter, loss.data.item()))
                # print(random_features.weight)
            iter += 1
            start += 1
            if iter > args.max_iter:
                break
        if iter > args.max_iter:
            break


    test_output = []
    start = 0
    node_list_length = len(node_list)
    while start * args.test_batch_size < node_list_length:
        batch_node_list = node_list[start * args.test_batch_size: (start+1) * args.test_batch_size]
        batch_output, _, _ = model.forward(torch.LongTensor(batch_node_list).to(device), None, None)
        test_batch_output = batch_output["nodes"]
        test_batch_output = test_batch_output.cpu().data.numpy()
        test_output.extend(test_batch_output)
        start += 1

    output_file = '{}/data/embedding/{}_cfane.txt'.format(args.prefix, args.dataset)
    print(output_file)
    with open(output_file, 'w') as f:
        print(len(test_output), len(test_output[0]), file=f)
        print('output dim', len(test_output), len(test_output[0]))
        for i, node_id in idx2id.items():
            print(node_id, end='', file=f)
            for item in test_output[i]:
                print(' {}'.format(item), file=f, end='')
            print('', file=f)
        f.close()

    print("Training completed")


def generate_context(walk, window_size):
    context = []
    cnt = 0
    for k in range(len(walk)):
        for i in range(len(walk[k])):
            for j in range(i+1, i+window_size+1):
                if j >= len(walk[k]):
                    break
                elif walk[k][i] == walk[k][j]:
                    cnt += 1
                else:
                    context.append((walk[k][i], walk[k][j]))
    print(cnt, "pairs removed")
    return context


def visualize(features, labels, title):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    features_2D = tsne.fit_transform(features)
    plt.scatter(features_2D[:, 0], features_2D[:, 1], c=labels, marker='.', cmap=plt.cm.rainbow)
    # plt.title(title)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./img/{}.png".format(args.dataset))


def main():
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)  # cpu
    print("Loading {} data...".format(args.dataset))
    train_data = load_data(args.prefix, args.dataset)
    print("Loading completed. Training starts")
    train(train_data)


if __name__ == "__main__":
    main()