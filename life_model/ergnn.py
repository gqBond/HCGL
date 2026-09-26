import torch
import copy
from .ergnn_utils import *
from .hero_utils import *
import pickle
from core.task_data import get_subgraph_func
import numpy as np

samplers = {'CM': CM_sampler(plus=False), 'CM_plus': CM_sampler(plus=True), 'MF': MF_sampler(plus=False),
            'MF_plus': MF_sampler(plus=True), 'random': random_sampler(plus=False)}


class NET(torch.nn.Module):
    """
        ER-GNN baseline for NCGL tasks

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

        """

    def __init__(self,
                 model,
                 args):
        super(NET, self).__init__()

        # setup network
        self.net = model
        self.args = args
        self.ergnn_args = args["ergnn_args"]
        self.sampler = samplers[self.ergnn_args['sampler'][0]]
        self.dataset = args["dataset"]
        self.n_tasks = args["num_tasks"]
        self.get_subgraph = get_subgraph_func(self.dataset)
        dataset_classes = {'DBLP': 4, 'Freebase': 7, 'IMDB': 5}
        self.n_classes = dataset_classes.get(self.dataset, 0)  # if dataset not in dict then return 0
        if self.n_classes == 0:
            raise ValueError("Dataset not exists!")
        self.n_labels_per_task = self.n_classes // self.n_tasks

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()
        # self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = []
        self.budget = int(self.ergnn_args['budget'][0])
        self.d_CM = self.ergnn_args['d'][0]  # d for CM sampler of ERGNN
        self.aux_g = None

    def forward(self, g, features):
        output, _ = self.net(g, features)
        return output

    def observe(self, g, features, labels, t, train_idx, ids_per_cls_train):
        """
                        The method for learning the given tasks under the task-IL setting.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class.
                        :param dataset: The entire dataset.

                        """
        if not isinstance(self.aux_g, list):  # make sure the type of aux_g: list
            self.aux_g = []
            self.buffer_node_ids = {}
            self.aux_loss_w_ = []  # store the weight of loss: beta = |B|/|B|+|D|

        n_nodes = len(train_idx)
        buffer_size = 0
        for k in self.buffer_node_ids:
            buffer_size += len(self.buffer_node_ids[k])
        beta = buffer_size / (buffer_size + n_nodes)

        self.net.train()
        self.net.zero_grad()
        logits, _ = self.net(g, features)
        # output_labels = labels[train_idx]

        loss = self.ce(logits[train_idx], labels[train_idx])

        if t != self.current_task:
            self.current_task = t
            if self.dataset == 'Freebase':
                sampled_ids = self.sampler(ids_per_cls_train, self.budget, features.to_dense(), self.net.second_last_h,
                                       self.d_CM)
                sub_g = get_sub_g(g, sampled_ids, features.to_dense(), labels, self.args['budget_ots'])
            else:
                sampled_ids = self.sampler(ids_per_cls_train, self.budget, features, self.net.second_last_h,
                                           self.d_CM)
                sub_g = get_sub_g(g, sampled_ids, features, labels, self.args['budget_ots'])
            # old_ids = g.nodes['0'].data['_ID']
            # self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
            # sub_g = self.get_subgraph(hg, self.buffer_node_ids[t], features_full, labels_full)
            self.buffer_node_ids[t] = sampled_ids
            self.aux_g.append(sub_g.to(self.args['device']))

        if t != 0:
            for tt in range(t):
                aux_g = self.aux_g[tt]
                # o1, o2 = tt * self.n_labels_per_task, (tt+1) * self.n_labels_per_task
                aux_features, aux_labels = aux_g.ndata['features']['0'], aux_g.ndata['labels']['0'].squeeze()
                if self.dataset == 'Freebase':
                    aux_features = aux_features.to_sparse()
                aux_features = aux_features.to(self.args['device'])
                aux_labels = aux_labels.to(self.args['device'])
                aux_logits, _ = self.net(aux_g, aux_features)
                loss_aux = self.ce(aux_logits, aux_labels)
                loss = beta * loss + (1 - beta) * loss_aux

        loss.backward()
        self.opt.step()

        return loss
