import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_gat import CustomGATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.requires_grad_(True)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        # Ensure beta is differentiable
        beta.retain_grad()

        # return semantic embedding and att_coefficient
        return (beta * z).sum(1), beta  # (N, D * K), (M, 1)

class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

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
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        all_attention_scores = []  # record all att_coef(meta path based)

        # print(f"Input feature shape: {h.shape}")

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embedding, attention_scores = self.gat_layers[i](new_g, h, get_attention=True)

            # # 这里确保 attention_scores 保持计算图信息
            # attention_scores = attention_scores.view(-1, 1)
            # attention_scores.retain_grad()

            # print(f"Meta-path {meta_path}: output shape after GATConv: {h.shape}")
            semantic_embeddings.append(semantic_embedding.flatten(1))  # shape:(N, out_size, K)->(N, out_size * K)
            all_attention_scores.append(attention_scores)

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        # print(f"Stacked semantic embeddings shape: {semantic_embeddings.shape}")

        h_out, semantic_attention_scores = self.semantic_attention(semantic_embeddings)
        # print(f"Output feature shape: {h_out.shape}")

        return h_out, all_attention_scores, semantic_attention_scores  # (N, D * K), attention scores


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
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
        logits = self.predict(h)
        return logits, all_attention_scores  # Return logits and attention scores


class HAN_freebase(nn.Module,):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN_freebase, self).__init__()

        self.layers = nn.ModuleList()
        self.fc = nn.Linear(in_size, hidden_size)
        self.layers.append(
            HANLayer(meta_paths, hidden_size, hidden_size, num_heads[0], dropout)
        )
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
        h = self.fc(h)
        h = torch.FloatTensor(h.cpu()).to(g.device)
        for i, gnn in enumerate(self.layers):
            h, attention_scores, semantic_attention_scores = gnn(g, h)
            all_attention_scores.append((attention_scores, semantic_attention_scores))
            if i == len(self.layers) - 2:
                self.second_last_h = h

        self.last_h = h
        logits = self.predict(h)
        return logits, all_attention_scores
