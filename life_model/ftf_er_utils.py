import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import dgl

try:
    from petsc4py import PETSc
except ImportError:
    PETSc = None


class Hodge_calculator(nn.Module):
    def __init__(self):
        super().__init__()

    def solve_petsc4py(self, A, b):
        ''' Solve Ax = b, where A is a 2D NumPy array and b is a 1D NumPy array. '''
        if PETSc is None:
            return self.solve_scipy(A, b)
        # Create PETSc matrices and vectors
        Amat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        bvec = PETSc.Vec().createWithArray(b)
        xvec = PETSc.Vec().createWithArray(np.zeros(b.shape[0]))  # Used to save the results

        # Initialize the KSP solver and set up the matrix
        ksp = PETSc.KSP().create()
        ksp.setOperators(Amat)

        # Set the solver type (optional)
        # ksp.setType('cg')  # Conjugate gradient method for symmetric positive definite matrices

        # Solve systems of linear equations
        ksp.solve(bvec, xvec)

        return xvec.array  # Convert the result into a NumPy array

    def solve_scipy(self, A, b):
        res = spsolve(A, b)
        return res

    # Create the Lv matrix
    def getLv(self, adj_matrix, cs):
        # Calculate the degree matrix
        cs[cs == 0] = 1  # 将孤立点度数设为1
        degree_matrix = sp.diags(cs, dtype=np.float32)

        # Calculate the Laplace matrix
        Lv = degree_matrix - adj_matrix

        # 防止非零值
        epsilon = 1e-6
        Lv.setdiag(Lv.diagonal() + epsilon)
        return Lv

    def forward(self, graph):
        graph = dgl.to_homogeneous(graph)
        indptr, indices, data = graph.adj_tensors('csr')

        # 确保张量在 CPU 上
        indptr_cpu = indptr.cpu().numpy()
        indices_cpu = indices.cpu().numpy()
        data_cpu = data.cpu().numpy()

        # 获取节点数量
        num_nodes = graph.num_nodes()

        # 转换为 SciPy 的 CSR 矩阵
        adj_matrix = sp.csr_matrix((data_cpu, indices_cpu, indptr_cpu), shape=(num_nodes, num_nodes))

        cs = adj_matrix.sum(0).A.flatten()  # 计算度向量
        Lv = self.getLv(adj_matrix, cs)

        # Solve systems of linear equations (sparse matrices)
        fai = self.solve_petsc4py(Lv, cs)

        # min-max normalization
        min_vals = np.min(fai) - 1e-2
        max_vals = np.max(fai)
        # min-max scaling, scale the range of x to [δ, scale], δ close to 0
        scaled_fai = (fai - min_vals) / (max_vals - min_vals)
        return scaled_fai


class loss_sampler(nn.Module):
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, g, features, labels, net, ce):
        socre_per_cls_train = self.cal_score(ids_per_cls_train, g, features, labels, net, ce)
        return self.sampling(ids_per_cls_train, budget, socre_per_cls_train)

    def sampling(self, ids_per_cls_train, budget, socre_per_cls_train):
        ids_selected = []
        for ids, scores in zip(ids_per_cls_train, socre_per_cls_train):
            sample_ids = np.argsort(scores)[::-1]  # The higher the score, the more priority
            sample = np.array(ids)[sample_ids]
            ids_selected.extend(sample[:budget])
        return ids_selected

    def cal_score(self, ids_per_cls_train, g, features, labels, net, ce):
        '''
        :param ids_per_cls_train: Indices of the nodes in each class.
        :param g: The graph of the last task.
        :param features: Node features of the last task.
        :param labels: Labels of the nodes in the last task.
        :param net: Model
        :param ce: Loss function.
        :return: scores
        '''

        socre_per_cls_train = []
        output, _ = net(g, features)
        if self.plus:  # Sampling by the gradient of loss
            for ids in ids_per_cls_train:
                scores = []
                for id in ids:
                    loss = ce(output[id], labels[id])
                    loss.backward(retain_graph=True)
                    grad = 0.0
                    for params in net.parameters():
                        grad += torch.norm(params.grad, p=1).cpu().item()
                    scores.append(grad)
                    net.zero_grad()
                socre_per_cls_train.append(scores)

        else:  # Sampling only by loss itself
            for ids in ids_per_cls_train:
                scores = ce(output[ids], labels[ids])
                scores = scores.cpu().tolist()
                socre_per_cls_train.append(scores)

        return socre_per_cls_train


class mix_sampler(nn.Module):
    def __init__(self, plus):
        super().__init__()
        self.plus = plus  # plus version for FTF-ER-prob. # normal version for FTF-ER-det.
        self.beta = 0.5  # beta in the paper
        self.feat_calculator = loss_sampler(True)

    def forward(self, ids_per_cls_train, budget, g, features, labels, net, ce, fai):
        score_feat_per_cls_train = self.feat_calculator.cal_score(ids_per_cls_train, g, features, labels, net, ce)
        score_topo_per_cls_train = []
        for ids in ids_per_cls_train:
            score_hodge = [-fai[id] for id in ids]
            score_topo_per_cls_train.append(score_hodge)
        socre_per_cls_train = []
        for score_loss, score_hodge in zip(score_feat_per_cls_train, score_topo_per_cls_train):
            score_loss_normalized = (score_loss - np.min(score_loss)) / (np.max(score_loss) - np.min(score_loss))
            score_hodge_normalized = (score_hodge - np.min(score_hodge)) / (np.max(score_hodge) - np.min(score_hodge))
            score = (1 - self.beta) * score_loss_normalized + self.beta * score_hodge_normalized
            score = score / np.sum(score)
            socre_per_cls_train.append(score)
        return self.sampling(ids_per_cls_train, budget, socre_per_cls_train)

    def sampling(self, ids_per_cls_train, budget, socre_per_cls_train):
        ids_selected = []
        for ids, scores in zip(ids_per_cls_train, socre_per_cls_train):
            if self.plus:
                scores[scores == 0] = 1e-6
                scores[np.isnan(scores)] = 1e-6
                scores = scores / np.sum(scores)
                sample = np.random.choice(ids, size=min(budget, len(ids)), p=scores, replace=False)
            else:
                sample_ids = np.argsort(scores)[::-1]
                sample = np.array(ids)[sample_ids]
                sample = sample[:budget]
            ids_selected.extend(sample)
        return ids_selected
