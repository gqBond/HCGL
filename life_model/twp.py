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
        self.fisher_loss = {}
        self.fisher_att = {}
        self.optpar = {}

        # Hyper-parameters
        self.lambda_l = args["lambda_l"]
        self.lambda_t = args["lambda_t"]
        self.beta = args["beta"]
        self.n_tasks = args["num_tasks"]

    def forward(self, g, features):
        logits, all_attention_scores = self.net(g, features)
        return logits, all_attention_scores

    def observe(self, g, features, labels, t, train_mask, dataset):
        self.net.train()

        # If a new task starts
        if t != self.current_task:
            self.net.zero_grad()
            self.fisher_loss[self.current_task] = []
            self.fisher_att[self.current_task] = []
            self.optpar[self.current_task] = []

            g_pre, features_pre, labels_pre, train_mask_pre, val_mask_pre, test_mask_pre = get_task_data(dataset,
                                                                                                         self.args, t)
            g_pre = g_pre.to(self.device)
            features_pre = features_pre.to(self.device)
            labels_pre = labels_pre.to(self.device)
            train_mask_pre = train_mask_pre.to(self.device)

            # Compute gradients for the previous task
            logits, all_attention_scores = self.net(g_pre, features_pre)  # use previous task's data
            loss = self.ce(logits[train_mask_pre], labels_pre[train_mask_pre])
            loss.backward(retain_graph=True)

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[self.current_task].append(pd)
                self.fisher_loss[self.current_task].append(pg)

            # Capture attention gradients for regularization
            for layer_idx, (attention_scores, semantic_attention_scores) in enumerate(all_attention_scores):
                # 对 GAT 层的注意力系数计算范数并反向传播
                for head_idx, att in enumerate(attention_scores):
                    eloss = torch.norm(att)
                    eloss.backward(retain_graph=True)

                eloss_semantic = torch.norm(semantic_attention_scores)
                eloss_semantic.backward(retain_graph=True)

            for p in self.net.parameters():
                # if p.grad is not None:
                pg = p.grad.data.clone().pow(2)
                self.fisher_att[self.current_task].append(pg)

            self.current_task = t

        self.net.zero_grad()
        logits, all_attention_scores = self.net(g, features)
        loss = self.ce(logits[train_mask], labels[train_mask])
        loss.backward(retain_graph=True)

        grad_norm = 0
        for p in self.net.parameters():
            pg = p.grad.data.clone()
            grad_norm += torch.norm(pg, p=1)

        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()

        loss = loss + self.beta * grad_norm
        loss.backward()
        self.opt.step()

        return loss
