import torch
import numpy as np
from core.task_data import get_task_data


class NET(torch.nn.Module):
    def __init__(self, model, args):
        super(NET, self).__init__()

        self.net = model
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
        self.ce = torch.nn.CrossEntropyLoss()
        self.device = args["device"]
        self.args = args

        # Initialize memories for Fisher Information Matrix and optimal parameters
        self.current_task = 0
        self.fisher = []
        self.optpar = []

        # Hyper-parameters
        # self.n_memories = args["n_memories"]
        self.n_tasks = args["num_tasks"]
        self.reg = args["memory_strength"]

    def forward(self, g, features):
        logits, all_attention_scores = self.net(g, features)
        return logits, all_attention_scores

    def observe(self, g, features, labels, t, train_mask, dataset):
        self.net.train()

        # If a new task starts
        if t != self.current_task:
            self.fisher = []
            self.optpar = []
            g_pre, features_pre, labels_pre, train_mask_pre, val_mask_pre, test_mask_pre = get_task_data(dataset,
                                                                                                         self.args, t)
            g_pre = g_pre.to(self.device)
            features_pre = features_pre.to(self.device)
            train_mask_pre = train_mask_pre.to(self.device)

            # Compute gradients for the previous task
            logits, all_attention_scores = self.net(g_pre, features_pre)  # use previous task's data
            logits = logits[train_mask_pre]
            logits.pow_(2)
            loss = logits.mean()
            self.net.zero_grad()
            loss.backward()

            # Save optimal parameters and compute Fisher Information Matrix
            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar.append(pd)
                self.fisher.append(pg)

            self.current_task = t

        self.net.zero_grad()
        logits, all_attention_scores = self.net(g, features)
        loss = self.ce(logits[train_mask], labels[train_mask])

        if t > 0:
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[i]
                l = l * (p - self.optpar[i]).pow(2)
                loss += l.sum()

        loss.backward()
        self.opt.step()

        return loss
