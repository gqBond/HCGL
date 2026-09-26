import torch
from .ftf_er_utils import *
from core.data_loader import get_label_ids
from core.task_data import get_subgraph_func
import os


def labels_mapping(task_id, labels, dataset, classes_per_task):
    label_ids = get_label_ids(dataset)
    task_classes = label_ids[
        task_id * classes_per_task:(task_id + 1) * classes_per_task
    ]
    mapping = {old_label: new_label for new_label, old_label in enumerate(task_classes)}
    return torch.LongTensor([mapping[label.item()] for label in labels])

calculators = {'Hodge': Hodge_calculator()}
samplers = {'mix': mix_sampler(plus=False), 'mix_plus': mix_sampler(plus=True)}


class NET(torch.nn.Module):
    """
        FTF-ER baseline for NCGL tasks

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.
        :param dataset: The entire dataset.

        """

    def __init__(self,
                 model,
                 hg,
                 args):
        super(NET, self).__init__()

        # setup network
        self.net = model
        self.ftfer_args = args['ftfer_args']
        self.sampler = samplers[self.ftfer_args['sampler'][0]]
        self.sampler.beta = self.ftfer_args['beta'][0]
        self.calculator = calculators['Hodge']
        self.args = args
        self.dataset = args["dataset"]
        self.n_tasks = args["num_tasks"]
        self.get_subgraph = get_subgraph_func(self.dataset)
        dataset_classes = {'DBLP': 4, 'Freebase': 7, 'IMDB': 5}
        self.n_classes = dataset_classes.get(self.dataset, 0)  # if dataset not in dict then return 0
        if self.n_classes == 0:
            raise ValueError("Dataset not exists!")
        self.n_cls_per_task = self.n_classes // self.n_tasks

        fai_path = os.path.join(args['data_path'], args['dataset'] + ".fai.npy")
        if os.path.exists(fai_path):
            self.fai = np.load(fai_path)
        else:
            self.fai = self.calculator(hg)
            np.save(fai_path, self.fai)

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = {}
        self.budget = self.ftfer_args['budget'][0]

    def forward(self, g, features):
        output, _ = self.net(g, features)
        return output

    def observe(self, g, hg, features_full, features, labels_full, labels, t, train_idx, ids_per_cls_train):
        """
            The method for learning the given tasks under the class-IL setting.

            :param g: The graph of the current task.
            :param features: Node features of the current task.
            :param labels: Labels of the nodes in the current task.
            :param t: Index of the current task.
            :param train_idx: The indices of the nodes participating in the training.
            :param ids_per_cls_train: Indices of the nodes in each class.
        """
        self.net.train()
        self.net.zero_grad()

        if t != self.current_task:
            if t > 0:
                last_task_sampled_ids = self.sampler(
                    self.last_task_ids_per_cls_train,
                    self.budget,
                    self.last_task_g,
                    self.last_features,
                    self.last_labels,
                    self.net,
                    self.ce,
                    self.fai
                )
                old_ids = self.last_task_g.nodes['0'].data['_ID']
                self.buffer_node_ids[self.current_task] = old_ids[last_task_sampled_ids].tolist()
            self.last_task_g = g
            self.last_task_ids_per_cls_train = ids_per_cls_train
            self.last_features = features
            self.last_labels = labels
            self.current_task = t

        if t == 0:
            logits, _ = self.net(g, features)
            loss = self.ce(logits[train_idx], labels[train_idx])
        else:
            nodes = []
            labels_list = []
            for tt in range(t):
                nodes.extend(self.buffer_node_ids[tt])
                labels_ = labels_mapping(tt, labels_full[self.buffer_node_ids[tt]], self.dataset,
                                         self.n_cls_per_task).to(self.args['device'])
                labels_list.append(labels_)
            olds_ids = g.nodes['0'].data['_ID']
            task_ids = olds_ids[train_idx]
            nodes.extend(task_ids.tolist())
            labels_list.append(labels[train_idx])
            mix_labels = torch.cat(labels_list, dim=0)
            mix_g = self.get_subgraph(hg, nodes, features_full, labels_full)
            mix_features = mix_g.ndata['features']['0']
            if self.dataset == 'Freebase':
                mix_features = mix_features.to_sparse()
            mix_features = mix_features.to(self.args['device'])
            logits, _ = self.net(mix_g, mix_features)
            loss = self.ce(logits, mix_labels)
        loss.backward()
        self.opt.step()

        return loss
