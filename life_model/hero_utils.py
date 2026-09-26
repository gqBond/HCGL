import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import copy
import dgl


class MF_sampler(nn.Module):
    # sampler for ERGNN MF and MF*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps)
        else:
            return self.sampling(ids_per_cls_train, budget, feats)

    def sampling(self, ids_per_cls_train, budget, vecs):
        centers = [vecs[ids].mean(0) for ids in ids_per_cls_train]
        sim = [centers[i].view(1, -1).mm(vecs[ids_per_cls_train[i]].permute(1, 0)).squeeze() for i in range(len(centers))]
        rank = [s.sort()[1].tolist() for s in sim]
        ids_selected = []
        for i, ids in enumerate(ids_per_cls_train):
            nearest = rank[i][0:min(budget, len(ids_per_cls_train[i]))]
            ids_selected.extend([ids[i] for i in nearest])
        return ids_selected


class CM_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d, using_half=True):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d, using_half=using_half)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, d, using_half=using_half)

    def sampling(self, ids_per_cls_train, budget, vecs, d, using_half=True):
        budget_dist_compute = 1000
        '''
        if using_half:
            vecs = vecs.half()
        '''
        vecs = vecs.half()
        ids_selected = []
        for i, ids in enumerate(ids_per_cls_train):  # 二维列表
            other_cls_ids = list(range(len(ids_per_cls_train)))
            other_cls_ids.pop(i)
            ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i]) < budget_dist_compute else random.choices(ids_per_cls_train[i], k=budget_dist_compute)

            dist = []
            vecs_0 = vecs[ids_selected0]
            for j in other_cls_ids:
                chosen_ids = random.choices(ids_per_cls_train[j], k=min(budget_dist_compute, len(ids_per_cls_train[j])))
                vecs_1 = vecs[chosen_ids]
                if len(chosen_ids) < 26 or len(ids_selected0) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0,vecs_1))

            #dist = [torch.cdist(vecs[ids_selected0], vecs[random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))]) for j in other_cls_ids]
            dist_ = torch.cat(dist, dim=-1)  # include distance to all the other classes
            n_selected = (dist_ < d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
            current_ids_selected = rank[:budget]
            ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
        return ids_selected


class CECM_sampler(nn.Module):
    # sampler for Class-wise Expected Coverage Minimization (C-ECM)
    # Based on Wasserstein bound improvement over CM
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d, using_half=True):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, using_half=using_half)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, using_half=using_half)

    def sampling(self, ids_per_cls_train, budget, vecs, using_half=True):
        """
        Class-wise Expected Coverage Minimization (C-ECM) sampling algorithm
        
        Args:
            ids_per_cls_train: List of lists, each sublist contains node IDs for a class
            budget: Number of samples to select per class
            vecs: Embedding vectors (features or representations) of shape (N, D)
            using_half: Whether to use half precision for computation
            
        Returns:
            ids_selected: List of selected node IDs
        """
        if using_half:
            vecs = vecs.half()
        else:
            vecs = vecs.float()
        
        ids_selected = []
        
        # Process each class separately
        for c, ids_c in enumerate(ids_per_cls_train):
            if len(ids_c) == 0:
                continue
                
            # Initialize S_c = ∅ (selected set for class c)
            S_c = []
            
            # Initialize d_c(x) = +∞ for all x in C_c
            # d_c stores the minimum distance from each candidate to the selected set
            d_c = torch.full((len(ids_c),), float('inf'), dtype=vecs.dtype, device=vecs.device)
            
            # Get embeddings for all candidates in this class
            vecs_c = vecs[ids_c]  # Shape: (|C_c|, D)
            num_candidates = len(ids_c)
            actual_budget = min(budget, num_candidates)
            
            # Precompute pairwise distance matrix if class size is reasonable
            # This avoids repeated distance computations
            use_precomputed = num_candidates <= 5000  # Threshold for memory efficiency
            if use_precomputed and num_candidates >= 26:
                # Precompute full pairwise distance matrix
                pairwise_distances = torch.cdist(vecs_c, vecs_c, p=2)  # Shape: (|C_c|, |C_c|)
            elif use_precomputed and num_candidates < 26:
                # For small tensors, use float precision
                pairwise_distances = torch.cdist(vecs_c.float(), vecs_c.float(), p=2).to(vecs.dtype)
            else:
                pairwise_distances = None  # Compute on-the-fly for large classes
            
            # Iteratively select samples
            for k in range(actual_budget):
                # Compute marginal gain for each candidate u in C_c \ S_c
                marginal_gains = []
                candidate_indices = []
                
                # Get remaining candidates (not yet selected)
                remaining_indices = [idx for idx in range(num_candidates) if idx not in S_c]
                
                if len(remaining_indices) == 0:
                    break
                
                for u_idx in remaining_indices:
                    candidate_indices.append(u_idx)
                    
                    # Get distances from u to all candidates
                    if pairwise_distances is not None:
                        # Use precomputed distances
                        distances_u = pairwise_distances[u_idx, :]  # Shape: (|C_c|,)
                    else:
                        # Compute on-the-fly for large classes
                        u_embedding = vecs_c[u_idx:u_idx+1]  # Shape: (1, D)
                        if num_candidates < 26:
                            distances_u = torch.cdist(
                                vecs_c.float(), 
                                u_embedding.float(), 
                                p=2
                            ).squeeze(-1).to(vecs.dtype)
                        else:
                            distances_u = torch.cdist(
                                vecs_c, 
                                u_embedding, 
                                p=2
                            ).squeeze(-1)  # Shape: (|C_c|,)
                    
                    # Compute marginal gain: Δ_c(u) = (1/|C_c|) * Σ_{x in C_c} max(0, d_c(x) - ||φ(x) - φ(u)||_2)
                    marginal_gain = torch.clamp(d_c - distances_u, min=0.0).sum() / num_candidates
                    marginal_gains.append(marginal_gain)
                
                if len(marginal_gains) == 0:
                    break
                
                # Select u* = argmax_u Δ_c(u)
                marginal_gains_tensor = torch.stack(marginal_gains)
                best_idx = torch.argmax(marginal_gains_tensor).item()
                u_star_idx = candidate_indices[best_idx]
                
                # Add u* to S_c
                S_c.append(u_star_idx)
                
                # Update d_c(x) = min(d_c(x), ||φ(x) - φ(u*)||_2) for all x in C_c
                if pairwise_distances is not None:
                    # Use precomputed distances
                    distances_to_u_star = pairwise_distances[u_star_idx, :]  # Shape: (|C_c|,)
                else:
                    # Compute on-the-fly
                    u_star_embedding = vecs_c[u_star_idx:u_star_idx+1]  # Shape: (1, D)
                    if num_candidates < 26:
                        distances_to_u_star = torch.cdist(
                            vecs_c.float(), 
                            u_star_embedding.float(), 
                            p=2
                        ).squeeze(-1).to(vecs.dtype)
                    else:
                        distances_to_u_star = torch.cdist(
                            vecs_c, 
                            u_star_embedding, 
                            p=2
                        ).squeeze(-1)  # Shape: (|C_c|,)
                
                d_c = torch.minimum(d_c, distances_to_u_star)
            
            # Add selected IDs for this class to the overall selection
            ids_selected.extend([ids_c[idx] for idx in S_c])
        
        return ids_selected


class random_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, d)

    def sampling(self,ids_per_cls_train, budget, vecs, d):
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            ids_selected.extend(random.sample(ids,min(budget,len(ids))))
        return ids_selected


class reservoir_sampler(nn.Module):
    def __init__(self, buffer_size):
        """
        初始化 Reservoir Sampler
        :param buffer_size: 缓冲区大小（采样节点的最大数量）
        """
        super().__init__()
        self.buffer_size = buffer_size
        self.buffer = []  # 用于存储采样的节点

    def forward(self, node_ids):
        """
        从节点流中采样节点
        :param node_ids: 当前任务的所有节点 ID 列表
        :return: 采样后的节点 ID 列表
        """
        for i, node_id in enumerate(node_ids):
            if len(self.buffer) < self.buffer_size:
                # 如果缓冲区未满，直接加入
                self.buffer.append(node_id)
            else:
                # 缓冲区已满，以概率 k / (i + 1) 替换缓冲区中的一个节点
                j = random.randint(0, i)
                if j < self.buffer_size:
                    self.buffer[j] = node_id
        return self.buffer


# 在每个类别上分别使用 Reservoir Sampling
def reservoir_sampling(node_ids_per_class, buffer_size):
    """
    对每个类别单独进行 Reservoir Sampling
    :param node_ids_per_class: 每个类别的节点 ID 列表（二维列表）
    :param buffer_size: 每个类别的缓冲区大小
    :return: 各类别采样后的节点 ID 列表
    """
    sampled_ids = []
    for class_ids in node_ids_per_class:
        sampler = reservoir_sampler(buffer_size)
        sampled_ids.extend(sampler(class_ids))
    return sampled_ids


class KMeans_sampler(nn.Module):
    def __init__(self, num_clusters):
        """
        K-Means 采样策略
        :param num_clusters: 质心的个数（K）
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.centroids = None  # 质心
        self.cluster_counts = None  # 记录每个质心的样本数

    def forward(self, buffer_size, features):
        """
        执行 K-Means 采样
        :param buffer_size: 经验缓冲区大小
        :param features: 节点特征 (N, D)
        :return: 采样得到的节点索引列表
        """
        features = features.cpu()
        num_nodes, feature_dim = features.shape

        # KMeans++ 初始化质心
        if self.centroids is None:
            initial_indices = torch.randperm(num_nodes)[:self.num_clusters]
            self.centroids = features[initial_indices].clone()

        # 初始化 cluster 计数器
        self.cluster_counts = torch.zeros(self.num_clusters)

        # 计算每个 cluster 的 buffer 大小
        buffer_size_per_cluster = ceil(buffer_size / self.num_clusters)

        # 初始化缓冲区
        buffer = {i: [] for i in range(self.num_clusters)}

        # 逐个样本进行 K-Means 归类
        for i in range(num_nodes):
            x = features[i]

            # 计算到所有质心的距离，并找到最近的质心
            distances = torch.norm(x - self.centroids, dim=1)
            j = torch.argmin(distances).item()  # 选择最近的质心

            # 更新质心
            self.cluster_counts[j] += 1
            self.centroids[j] += (x - self.centroids[j]) / self.cluster_counts[j]

            # 计算到质心的距离
            d = distances[j].item()

            # 存储更接近质心的样本
            if len(buffer[j]) < buffer_size_per_cluster:
                buffer[j].append((i, d))  # 直接添加
            else:
                # 替换最远的样本
                max_dist_idx = max(range(len(buffer[j])), key=lambda idx: buffer[j][idx][1])
                if d < buffer[j][max_dist_idx][1]:  # 仅在更接近质心时替换
                    buffer[j][max_dist_idx] = (i, d)

        # 收集最终的采样结果
        selected_nodes = [node[0] for cluster_nodes in buffer.values() for node in cluster_nodes]

        return selected_nodes


class CBKMSampler:
    def __init__(self, total_buffer_size):
        """
        Class Balanced K-Means Sampling (CBKMS)

        :param total_buffer_size: 经验回放的总缓冲区大小
        """
        self.total_buffer_size = total_buffer_size

    def sample(self, ids_per_cls, features):
        """
        执行 CBKMS 采样策略
        :param ids_per_cls: 列表，每个子列表包含属于同一类别的节点索引，例如 [[id1, id2, ...], [id3, id4, ...]]
        :param features: 所有节点的特征矩阵 (N, D)
        :return: 采样得到的节点索引列表
        """
        features = features.cpu()
        sampled_nodes = []
        num_classes = len(ids_per_cls)

        # 计算每个类别的样本权重 ω_j
        class_sizes = [len(ids) for ids in ids_per_cls]
        total_samples = sum(class_sizes)
        class_weights = [size / total_samples for size in class_sizes]

        # 计算每个类别的缓冲区大小 B_j
        buffer_sizes = [int(self.total_buffer_size * weight) for weight in class_weights]

        # 遍历每个类别
        for class_idx, ids in enumerate(ids_per_cls):
            if len(ids) == 0:
                continue

            # 选择初始质心
            centroid_id = random.choice(ids)
            centroid = features[centroid_id].clone()
            buffer = []
            num_selected = 0

            for node_id in ids:
                # 计算当前节点到质心的欧几里得距离
                node_feature = features[node_id]
                dist = torch.norm(node_feature - centroid, p=2).item()

                # 更新质心
                num_selected += 1
                centroid += (node_feature - centroid) / num_selected

                # 维护缓冲区
                if len(buffer) < buffer_sizes[class_idx]:
                    buffer.append((node_id, dist))
                else:
                    # 计算最远点
                    farthest_node, max_dist = max(buffer, key=lambda x: x[1])

                    # 计算替换概率 p_r
                    replace_prob = max_dist / (dist + max_dist)

                    # 以 p_r 概率替换最远的点
                    if random.random() < replace_prob:
                        buffer.remove((farthest_node, max_dist))
                        buffer.append((node_id, dist))

            # 记录采样结果
            sampled_nodes.extend([node[0] for node in buffer])

        return sampled_nodes


class TopologyReconstruction(nn.Module):
    def __init__(self, threshold=0.1, mode='cosine', device=None):
        super(TopologyReconstruction, self).__init__()
        self.threshold = threshold
        self.mode = mode
        self.device = device

    def compute_adj(self, h):
        # generate adj matrix based on feature

        if self.mode == 'cosine':
            sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
        elif self.mode == 'dot':
            sim = torch.matmul(h, h.t())
        adj = torch.where(sim < self.threshold, torch.zeros_like(sim), sim)
        # mask = sim < self.threshold
        # adj = sim * mask
        adj = torch.sigmoid(adj)  # 映射到[0,1]
        return adj

    def forward(self, old_h, new_h):
        old_adj = self.compute_adj(old_h)
        new_adj = self.compute_adj(new_h)
        loss = torch.norm(new_adj - old_adj, 2)
        return loss


def get_sub_g(hg, target_nodes, features, labels, budget):
    target_nodes = copy.deepcopy(target_nodes)
    target_ntype = '0'
    # related_nodes = {ntype: set() for ntype in hg.ntypes if ntype != target_ntype}
    related_nodes = {}
    for ntype in hg.ntypes:
        if ntype != target_ntype:
            nodes = hg.nodes(ntype)
            perm_idx = torch.randperm(len(nodes))
            nodes_shuffle = nodes[perm_idx]
            related_nodes[ntype] = nodes_shuffle[:min(budget, len(nodes_shuffle))]

    node_ids_per_type = {ntype: list(nodes) for ntype, nodes in related_nodes.items()}
    if not isinstance(target_nodes, list):
        target_nodes = target_nodes.tolist()
    node_ids_per_type[target_ntype] = target_nodes

    subgraph = dgl.node_subgraph(hg, node_ids_per_type)
    subgraph.ndata['features'] = {
        '0': features[target_nodes]
    }
    subgraph.ndata['labels'] = {
        '0': labels[target_nodes]
    }
    return subgraph


def topk_nodes_by_degree(g, nodes, edges, k, direction='out'):
    degree = torch.zeros(len(nodes)).to(g.device)
    nodes = nodes.to(g.device)

    for etype in edges:
        if direction == 'out':
            deg = g.out_degrees(nodes, etype=etype)
        elif direction == 'in':
            deg = g.in_degrees(nodes, etype=etype)
        else:
            raise ValueError("direction must be 'in' or 'out'")
        degree += deg

    # sorted by degrees and select top k nodes
    k = min(k, len(degree))
    topk_deg, topk_idx = torch.topk(degree, k)
    topk_nodes = nodes[topk_idx]

    return topk_nodes


def sub_g_sampling_dblp(hg, target_nodes, features, labels, budget):
    target_nodes = copy.deepcopy(target_nodes)
    # 获取相关节点（根据边类型）
    related_p_nodes = set()
    related_c_nodes = set()
    related_t_nodes = set()

    for a_node in target_nodes:
        # 获取 `author` -> `publication` (`ap`)
        if 'ap' in hg.etypes:
            related_p_nodes.update(hg.successors(a_node, etype='ap').tolist())

        # 获取 `author` <- `publication` (`pa`)
        if 'pa' in hg.etypes:
            related_p_nodes.update(hg.predecessors(a_node, etype='pa').tolist())

    for p_node in related_p_nodes:
        # 获取 `publication` -> `conference` (`pc`)
        if 'pc' in hg.etypes:
            related_c_nodes.update(hg.successors(p_node, etype='pc').tolist())

        # 获取 `publication` -> `term` (`pt`)
        if 'pt' in hg.etypes:
            related_t_nodes.update(hg.successors(p_node, etype='pt').tolist())

        # 获取 `publication` <- `conference` (`cp`)
        if 'cp' in hg.etypes:
            related_c_nodes.update(hg.predecessors(p_node, etype='cp').tolist())

        # 获取 `publication` <- `term` (`tp`)
        if 'tp' in hg.etypes:
            related_t_nodes.update(hg.predecessors(p_node, etype='tp').tolist())

    if not isinstance(target_nodes, list):
        target_nodes = target_nodes.tolist()

    related_p_nodes = torch.tensor(list(related_p_nodes))
    related_c_nodes = torch.tensor(list(related_c_nodes))
    related_t_nodes = torch.tensor(list(related_t_nodes))
    p_edges = ['pa', 'pc', 'pt']
    c_edge = ['cp']
    t_edge = ['tp']
    p_nodes = topk_nodes_by_degree(hg, related_p_nodes, p_edges, budget)
    c_nodes = topk_nodes_by_degree(hg, related_c_nodes, c_edge, budget)
    t_nodes = topk_nodes_by_degree(hg, related_t_nodes, t_edge, budget)

    # 定义每种节点类型需要保留的节点ID
    node_ids_per_type = {
        '0': target_nodes,
        '1': list(p_nodes),
        '2': list(c_nodes),
        '3': list(t_nodes)
    }

    # 构建子图
    subgraph = dgl.node_subgraph(hg, node_ids_per_type)
    subgraph.ndata['features'] = {
        '0': features[target_nodes]
    }
    subgraph.ndata['labels'] = {
        '0': labels[target_nodes]
    }

    return subgraph


def sub_g_sampling_imdb(hg, target_nodes, features, labels, budget):
    target_nodes = copy.deepcopy(target_nodes)
    # 获取相关`director`（`d`）、`actor`（`a`）和`keyword`（`k`）节点
    related_d_nodes = set()
    related_a_nodes = set()
    related_k_nodes = set()
    for m_node in target_nodes:
        for etype in ['md', 'dm']:
            if etype in hg.etypes:
                if etype == 'md':
                    related_nodes = hg.successors(m_node, etype=etype)
                else:
                    related_nodes = hg.predecessors(m_node, etype=etype)
                related_d_nodes.update(related_nodes.tolist())

        for etype in ['ma', 'am']:
            if etype in hg.etypes:
                if etype == 'ma':
                    related_nodes = hg.successors(m_node, etype=etype)
                else:
                    related_nodes = hg.predecessors(m_node, etype=etype)
                related_a_nodes.update(related_nodes.tolist())

        for etype in ['mk', 'km']:
            if etype in hg.etypes:
                if etype == 'mk':
                    related_nodes = hg.successors(m_node, etype=etype)
                else:
                    related_nodes = hg.predecessors(m_node, etype=etype)
                related_k_nodes.update(related_nodes.tolist())

    if not isinstance(target_nodes, list):
        target_nodes = target_nodes.tolist()

    related_d_nodes = torch.tensor(list(related_d_nodes))
    related_k_nodes = torch.tensor(list(related_k_nodes))
    related_a_nodes = torch.tensor(list(related_a_nodes))
    d_edge = ['dm']
    k_edge = ['km']
    a_edge = ['am']
    d_nodes = topk_nodes_by_degree(hg, related_d_nodes, d_edge, budget)
    k_nodes = topk_nodes_by_degree(hg, related_k_nodes, k_edge, budget)
    a_nodes = topk_nodes_by_degree(hg, related_a_nodes, a_edge, budget)

    node_ids_per_type = {
        '0': target_nodes,
        '1': list(d_nodes),
        '2': list(a_nodes),
        '3': list(k_nodes)
    }

    # 构建子图
    subgraph = dgl.node_subgraph(hg, node_ids_per_type)
    subgraph.ndata['features'] = {
        '0': features[target_nodes]
    }
    subgraph.ndata['labels'] = {
        '0': labels[target_nodes]
    }

    return subgraph


def sub_g_sampling_freebase(hg, target_nodes, features, labels, budget):
    target_nodes = copy.deepcopy(target_nodes)
    target_ntype = '0'
    related_nodes = {ntype: set() for ntype in hg.ntypes if ntype != target_ntype}

    for target_node in target_nodes:
        for etype in ['01', '10']:
            if etype in hg.etypes:
                if etype == '01':
                    related_nodes1 = hg.successors(target_node, etype=etype)
                else:
                    related_nodes1 = hg.predecessors(target_node, etype=etype)
                related_nodes['1'].update(related_nodes1.tolist())

        for etype in ['04', '40']:
            if etype in hg.etypes:
                if etype == '04':
                    related_nodes4 = hg.successors(target_node, etype=etype)
                else:
                    related_nodes4 = hg.predecessors(target_node, etype=etype)
                related_nodes['4'].update(related_nodes4.tolist())

        for etype in ['07', '70']:
            if etype in hg.etypes:
                if etype == '07':
                    related_nodes7 = hg.successors(target_node, etype=etype)
                else:
                    related_nodes7 = hg.predecessors(target_node, etype=etype)
                related_nodes['7'].update(related_nodes7.tolist())

        if '05' in hg.etypes:
            related_nodes5 = hg.successors(target_node, etype='05')
            related_nodes['5'].update(related_nodes5.tolist())

        if '06' in hg.etypes:
            related_nodes6 = hg.successors(target_node, etype='06')
            related_nodes['6'].update(related_nodes6.tolist())

        if '20' in hg.etypes:
            related_nodes2 = hg.predecessors(target_node, etype='20')
            related_nodes['2'].update(related_nodes2.tolist())

        if '30' in hg.etypes:
            related_nodes3 = hg.predecessors(target_node, etype='30')
            related_nodes['3'].update(related_nodes3.tolist())

    # 定义每种节点类型需要保留的节点ID
    node_ids_per_type = {ntype: list(nodes) for ntype, nodes in related_nodes.items()}
    if not isinstance(target_nodes, list):
        target_nodes = target_nodes.tolist()
    selected_node_ids = {}
    selected_node_ids[target_ntype] = target_nodes
    etype_dict = {
        '1': ['10'],
        '2': ['20'],
        '3': ['30'],
        '4': ['40'],
        '5': ['05'],
        '6': ['06'],
        '7': ['70'],
    }
    for ntype, nodes_ids in node_ids_per_type.items():
        nodes_ids = torch.tensor(nodes_ids, dtype=torch.int64)
        if ntype in ['5', '6']:
            selected_nodes = topk_nodes_by_degree(hg, nodes_ids, etype_dict[ntype], budget, direction='in')
        else:
            selected_nodes = topk_nodes_by_degree(hg, nodes_ids, etype_dict[ntype], budget)
        selected_node_ids[ntype] = selected_nodes.tolist()

    subgraph = dgl.node_subgraph(hg, selected_node_ids)
    subgraph.ndata['features'] = {
        '0': features[target_nodes]
    }
    subgraph.ndata['labels'] = {
        '0': labels[target_nodes]
    }

    return subgraph


def sub_g_sampling(dataset):
    get_subgraph = None
    if dataset == 'Freebase':
        get_subgraph = sub_g_sampling_freebase
    elif dataset == 'IMDB':
        get_subgraph = sub_g_sampling_imdb
    elif dataset == 'DBLP':
        get_subgraph = sub_g_sampling_dblp
    else:
        raise ValueError("Name of Dataset is not correct!")

    return get_subgraph
