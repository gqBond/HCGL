import torch
from .ergnn_utils import *
from core.data_loader import get_label_ids
from core.task_data import get_subgraph_func


def labels_mapping(task_id, labels, dataset, classes_per_task):
    label_ids = get_label_ids(dataset)
    task_classes = label_ids[
        task_id * classes_per_task:(task_id + 1) * classes_per_task
    ]
    mapping = {old_label: new_label for new_label, old_label in enumerate(task_classes)}
    return torch.LongTensor([mapping[label.item()] for label in labels])

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
        self.ergnn_args = args["ergnn_args"]
        self.sampler = samplers[self.ergnn_args['sampler'][0]]
        self.dataset = args["dataset"]
        self.n_tasks = args["num_tasks"]
        self.get_subgraph = get_subgraph_func(self.dataset)
        self.args = args
        dataset_classes = {'DBLP': 4, 'Freebase': 7, 'IMDB': 5}
        self.n_classes = dataset_classes.get(self.dataset, 0)  # if dataset not in dict then return 0
        if self.n_classes == 0:
            raise ValueError("Dataset not exists!")
        self.n_labels_per_task = self.n_classes // self.n_tasks

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

        # setup losses
        # self.ce = torch.nn.CrossEntropyLoss()
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = {}
        self.budget = int(self.ergnn_args['budget'][0])
        self.d_CM = self.ergnn_args['d'][0]  # d for CM sampler of ERGNN
        self.aux_g = []

        # lamaml
        self.alpha_lr = nn.ParameterList([])
        for p in self.net.parameters():
            self.alpha_lr.append(
                nn.Parameter(5e-3 * torch.ones(p.shape, requires_grad=True).to(args["device"])))
        self.opt_lr = torch.optim.Adam(self.alpha_lr, lr=args["lr"], weight_decay=args["weight_decay"])
        self.ep = 0

    def forward(self, g, features):
        output, _ = self.net(g, features)
        return output

    def apply_fast_weights(self, model, fast_weights):
        # update params
        with torch.no_grad():
            for param, fast_param in zip(model.parameters(), fast_weights):
                param.copy_(fast_param)

    def inner_update(self, g, features, train_idx, fast_weights, labels):
        if fast_weights is None:
            fast_weights = list(map(lambda p: p, self.net.parameters()))

        self.apply_fast_weights(self.net, fast_weights)
        logits, _ = self.net(g, features)
        loss = self.ce(logits[train_idx], labels[train_idx])
        grads = list(torch.autograd.grad(loss, fast_weights))

        for i in range(len(grads)):
            grads[i] = torch.clamp(grads[i], -0.15, 0.15)
        # fast_weights = list(map(lambda p: p[1][0] - nn.functional.leaky_relu(p[1][1], negative_slope=0.2) * p[0], zip(grads, zip(fast_weights, self.alpha_lr))))
        if self.ep == 1:
            self.first_moment = list(map(lambda p: torch.zeros_like(p), grads))
            self.second_moment = list(map(lambda p: torch.zeros_like(p), grads))
        self.first_moment = list(map(lambda p: 0.9 * p[0] + (1 - 0.9) * p[1], zip(self.first_moment, grads)))
        self.second_moment = list(map(lambda p: 0.99 * p[0] + (1 - 0.99) * (p[1] ** 2), zip(self.second_moment, grads)))
        self.first_moment = [p / (1 - 0.9 ** (self.ep)) for p in self.first_moment]
        self.second_moment = [p / (1 - 0.99 ** (self.ep)) for p in self.second_moment]
        fast_weights = list(map(lambda p: p[0] - nn.functional.leaky_relu(p[1][2], negative_slope=0.2) * (
                    p[1][0] / (torch.sqrt(p[1][1] + 1e-8))),
                                zip(fast_weights, zip(self.first_moment, self.second_moment, self.alpha_lr))))  # adam
        return fast_weights

    def meta_loss(self, g, features, train_idx, fast_weights, labels):
        self.apply_fast_weights(self.net, fast_weights)
        logits, _ = self.net(g, features)
        loss = self.ce(logits[train_idx], labels[train_idx])
        return loss, logits

    def meta_loss_(self, g, features, fast_weights, labels):
        self.apply_fast_weights(self.net, fast_weights)
        logits, _ = self.net(g, features)
        loss = self.ce(logits, labels)
        return loss, logits

    def observe(self, g, hg, features_full, features, labels_full, labels, t, train_idx, ids_per_cls_train):
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
        self.net.train()
        if t != self.current_task:
            self.current_task = t
        self.ep += 1
        self.net.zero_grad()
        fast_weights = None

        if t == 0:
            fast_weights = self.inner_update(g, features, train_idx, fast_weights, labels)
            meta_loss, logits = self.meta_loss(g, features, train_idx, fast_weights, labels)
            self.net.zero_grad()
            self.alpha_lr.zero_grad()
            meta_loss.backward()
        else:
            nodes = []
            labels_list = []
            for tt in range(t):
                nodes.extend(self.buffer_node_ids[tt])
                labels_ = labels_mapping(tt, labels_full[self.buffer_node_ids[tt]], self.dataset,
                                         self.n_labels_per_task).to(self.args['device'])
                labels_list.append(labels_)
            olds_ids = g.nodes['0'].data['_ID']
            task_ids = olds_ids[train_idx]
            nodes.extend(task_ids.tolist())
            labels_list.append(labels[train_idx])
            aux_labels = torch.cat(labels_list, dim=0)
            aux_g = self.get_subgraph(hg, nodes, features_full, labels_full)
            aux_features = aux_g.ndata['features']['0']
            if self.dataset == 'Freebase':
                aux_features = aux_features.to_sparse()
            aux_features = aux_features.to(self.args['device'])

            fast_weights = self.inner_update(g, features, train_idx, fast_weights, labels)
            meta_loss, logits = self.meta_loss_(aux_g, aux_features, fast_weights, aux_labels)

            self.net.zero_grad()
            self.alpha_lr.zero_grad()
            meta_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2)
        torch.nn.utils.clip_grad_norm_(self.alpha_lr.parameters(), 2)
        self.opt.step()
        self.opt_lr.step()
        if self.ep == self.args["num_epochs"]:
            self.ep = 0
            if self.dataset == 'Freebase':
                sampled_nodes = self.sampler(ids_per_cls_train, self.budget, features.to_dense(), self.net.second_last_h,
                                         self.d_CM)
            else:
                sampled_nodes = self.sampler(ids_per_cls_train, self.budget, features, self.net.second_last_h,
                                             self.d_CM)
            old_ids = g.nodes['0'].data['_ID']
            self.buffer_node_ids[t] = old_ids[sampled_nodes].tolist()
        return meta_loss
