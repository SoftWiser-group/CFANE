import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class MultiviewModel(nn.Module):

    def __init__(self, final_dim, features, last_agg, autoencoder, neg_weight=7, loss_fn="skipgram",
                 view_selection=True, cross_fusion=True, output_norm=True):
        super(MultiviewModel, self).__init__()

        self.features = features
        self.view_output = last_agg
        self.autoencoder = autoencoder
        self.agg_weight = nn.Linear(last_agg.output_dim, final_dim, bias=False)
        self.ff_weight = nn.Linear(last_agg.output_dim, final_dim, bias=False)

        self.mu = nn.Linear(final_dim, 1, bias=False)

        self.neg_weight = neg_weight
        self.recons_criterion = nn.MSELoss(reduction='mean')

        if loss_fn == "skipgram":
            self.loss_fn = self.skip_gram
        else:
            """default"""
            self.loss_fn = self.skip_gram

        self.view_selection = view_selection
        self.cross_fusion = cross_fusion
        self.output_norm = output_norm



    def forward(self, nodes, contexts, neg_samples, in_drop=0.0):
        output = {}
        """"nodes"""
        if not nodes is None:
            nodes_agg_output, nodes_ff_output = self.view_output(nodes, in_drop)

            nodes_agg_output = self.agg_weight(nodes_agg_output)
            nodes_ff_output = self.ff_weight(nodes_ff_output)

            if self.view_selection:
                if self.output_norm:
                    nodes_agg_output = F.normalize(nodes_agg_output)
                    nodes_ff_output = F.normalize(nodes_ff_output)

                nodes_output = torch.stack((nodes_agg_output, nodes_ff_output), dim=1)
                nodes_coef = F.softmax(self.mu(nodes_output), dim=1)
                nodes_output = torch.sum(nodes_coef * nodes_output, dim=1)
            else:
                nodes_output = torch.cat((nodes_ff_output, nodes_agg_output), dim=1)
        else:
            nodes_output = None
        output.update({"nodes": nodes_output, "nodes_agg": nodes_agg_output, "nodes_ff": nodes_ff_output})

        """context nodes"""
        if not contexts is None:
            contexts_agg_output, contexts_ff_output = self.view_output(contexts, in_drop)

            contexts_agg_output = self.agg_weight(contexts_agg_output)
            contexts_ff_output = self.ff_weight(contexts_ff_output)

            if self.view_selection:
                if self.output_norm:
                    contexts_agg_output = F.normalize(contexts_agg_output)
                    contexts_ff_output = F.normalize(contexts_ff_output)

                contexts_output = torch.stack((contexts_agg_output, contexts_ff_output), dim=1)
                contexts_coef = F.softmax(self.mu(contexts_output), dim=1)
                contexts_output = torch.sum(contexts_coef * contexts_output, dim=1)
            else:
                contexts_output = torch.cat((contexts_ff_output, contexts_agg_output), dim=1)

        else:
            contexts_output = None
        output.update({"contexts": contexts_output})

        """negative samples"""
        if not neg_samples is None:
            neg_agg_output, neg_ff_output = self.view_output(neg_samples, in_drop)

            neg_agg_output = self.agg_weight(neg_agg_output)
            neg_ff_output = self.ff_weight(neg_ff_output)

            if self.view_selection:
                if self.output_norm:
                    neg_agg_output = F.normalize(neg_agg_output)
                    neg_ff_output = F.normalize(neg_ff_output)

                neg_output = torch.stack((neg_agg_output, neg_ff_output), dim=1)
                neg_coef = F.softmax(self.mu(neg_output), dim=1)
                neg_output = torch.sum(neg_coef * neg_output, dim=1)
            else:
                neg_output = torch.cat((neg_ff_output, neg_agg_output), dim=1)

        else:
            neg_output = None
        output.update({"neg": neg_output})

        # nodes_agg_attn_coef, nodes_ff_attn_coef = torch.unbind(nodes_coef, dim=1)
        nodes_agg_attn_coef, nodes_ff_attn_coef = None, None
        return output, nodes_agg_attn_coef, nodes_ff_attn_coef

    def loss(self, nodes, contexts, neg_samples, in_drop=0.0, task=0, balance=2):
        output, _, _ = self.forward(nodes, contexts, neg_samples, in_drop)
        nodes_output = output["nodes"]
        nodes_agg_output = output["nodes_agg"]
        nodes_ff_output = output["nodes_ff"]
        contexts_output = output["contexts"]
        neg_output = output["neg"]

        feats = self.features(nodes)
        recons_feats = self.autoencoder.decode(nodes_output)

        contexts_feats = self.features(contexts)
        recons_contexts_feats = self.autoencoder.decode(contexts_output)

        if self.cross_fusion == False or self.view_selection == False:
            reg_loss = 0
        else:
            reg_loss = 1e3 * torch.pow(torch.mean(torch.pow(nodes_agg_output - nodes_ff_output, 2)) - 0.0005, 2)

        if task == 0:
            skipgram_loss = self.loss_fn(nodes_output, contexts_output, neg_output)
            recons_loss = balance * (self.recons_criterion(feats, recons_feats) + self.recons_criterion(contexts_feats,
                                                                                       recons_contexts_feats))
        elif task == 1:
            skipgram_loss = 0
            recons_loss = balance * (self.recons_criterion(feats, recons_feats) + self.recons_criterion(contexts_feats,
                                                                                                    recons_contexts_feats))
        else:
            skipgram_loss = self.loss_fn(nodes_output, contexts_output, neg_output)
            recons_loss = 0
        # print("skip_gram loss", skipgram_loss)
        # print("recons loss", recons_loss)
        return recons_loss + skipgram_loss + reg_loss

    def skip_gram(self, nodes_output, neighbors_output, neg_output):
        aff = torch.mean(-torch.log(torch.sigmoid(torch.sum((nodes_output * neighbors_output), dim=1))))
        neg_aff = torch.mean(-torch.log(torch.sigmoid(-torch.mm(nodes_output, neg_output.t()))))
        loss = self.neg_weight * neg_aff + aff
        return loss
