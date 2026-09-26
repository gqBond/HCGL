"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from .custom_gat import CustomGATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = F.sigmoid(w)  # (M, 1)
        beta = beta.requires_grad_(True)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        # Ensure beta is differentiable
        beta.retain_grad()

        return (beta * z).sum(1), beta  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                CustomGATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True)
            )
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        all_attention_scores = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                temp_graph = dgl.metapath_reachable_graph(g, meta_path)
                self._cached_coalesced_graph[meta_path] = temp_graph

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embedding, attention_scores = self.gat_layers[i](new_g, h, get_attention=True)
            semantic_embeddings.append(semantic_embedding.flatten(1))
            all_attention_scores.append(attention_scores)

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        h_out, semantic_attention_scores = self.semantic_attention(semantic_embeddings)

        return h_out, all_attention_scores, semantic_attention_scores  # (N, D * K)


class HAN_IMDB(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN_IMDB, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout)
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)
        self.second_last_h = None
        self.last_h = None

    def forward(self, g, h):
        all_attention_scores = []
        for i, gnn in enumerate(self.layers):
            h, attention_scores, semantic_attention_scores = gnn(g, h)
            all_attention_scores.append((attention_scores, semantic_attention_scores))
            if i == len(self.layers) - 2:
                self.second_last_h = h

        self.last_h = h
        return self.predict(h), all_attention_scores


class HAN_freebase(nn.Module,):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN_freebase, self).__init__()

        self.layers = nn.ModuleList()
        self.fc = nn.Linear(in_size, hidden_size)
        self.layers.append(HANLayer(meta_paths, hidden_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout)
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)
        self.second_last_h = None

    def forward(self, g, h):
        all_attention_scores = []
        h = self.fc(h)
        h = torch.FloatTensor(h.cpu())
        for i, gnn in enumerate(self.layers):
            h, attention_scores, semantic_attention_scores = gnn(g, h)
            all_attention_scores.append((attention_scores, semantic_attention_scores))
            if i == len(self.layers) - 2:
                self.second_last_h = h

        logits = self.predict(h)
        return logits, all_attention_scores
