import torch
import torch.nn.functional as F
import copy
from .hero_utils import *
import pickle
import numpy as np
import time

samplers = {'CM': CM_sampler(plus=False), 'CM_plus': CM_sampler(plus=True), 'MF': MF_sampler(plus=False),
            'MF_plus': MF_sampler(plus=True), 'random': random_sampler(plus=False),
            'CECM': CECM_sampler(plus=False), 'CECM_plus': CECM_sampler(plus=True)}


class NET(torch.nn.Module):

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
        self.get_subgraph = sub_g_sampling(self.dataset)
        self.args = args

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
        self.inner_lr = args["lr"]

        self.ce = torch.nn.CrossEntropyLoss()
        # self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = {}
        self.budget = args["shots"]
        self.d_CM = self.ergnn_args['d'][0] 
        self.aux_g = []
        self.kt = args["k_teachers"]
        # kd
        self.teachers = []
        self.topo_recons = TopologyReconstruction()

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

    def distillation_loss(self, student_logits, teacher_logits, temperature=1.0):
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        return loss

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
        logits, att_scores = self.net(g, features)
        loss = self.ce(logits[train_idx], labels[train_idx])
        return loss, logits, att_scores

    def observe(self, g, features, labels, t, train_idx, ids_per_cls):

        self.net.train()
        self.net.zero_grad()
        fast_weights = None
        
        if t != self.current_task:
            self.ep = 0
            self.current_task = t
            if self.args['dataset'] == 'Freebase':
                sampled_nodes = self.sampler(ids_per_cls, self.budget, features.to_dense(), self.net.second_last_h,
                                             self.d_CM)
                sub_g = self.get_subgraph(g, sampled_nodes, features.to_dense(), labels, self.args['budget_ots'])
            else:
                sampled_nodes = self.sampler(ids_per_cls, self.budget, features, self.net.second_last_h, self.d_CM)
                sub_g = self.get_subgraph(g, sampled_nodes, features, labels, self.args['budget_ots'])
            self.buffer_node_ids[t] = sampled_nodes
            self.aux_g.append(sub_g)
            self.teachers.append(copy.deepcopy(self.net).to(self.args['device']))
            
        self.ep += 1
        fast_weights = self.inner_update(g, features, self.buffer_node_ids[t], fast_weights, labels)
        meta_loss, logits, att_scores = self.meta_loss(g, features, train_idx, fast_weights, labels)
        

        if t != 0:
            loss_er = 0.0
            for tt in range(t):
                aux_g = self.aux_g[tt]
                aux_features, aux_labels = aux_g.ndata['features']['0'], aux_g.ndata['labels']['0'].squeeze()
                if self.dataset == 'Freebase':
                    aux_features = aux_features.to_sparse()
                aux_features = aux_features.to(self.args['device'])
                aux_labels = aux_labels.to(self.args['device'])
                aux_logits, _ = self.net(aux_g, aux_features)
                loss_aux = self.ce(aux_logits, aux_labels)
                loss_er += loss_aux
            
            
            kd_node, kd_sem = 0.0, 0.0
            start_i = max(1, len(self.teachers) - self.kt)
            for i in range(start_i, len(self.teachers)):  # use all previous teachers model
                teacher_model = self.teachers[i]
                # teacher logits
                with torch.no_grad():
                    teacher_logits, att_scores_t = teacher_model(g, features)

                # node-level
                distill_loss = self.distillation_loss(logits, teacher_logits, self.args["temp"])
                kd_node += distill_loss

                # semantic-level
                _, sem_att_score_pre = att_scores_t[-1]
                _, sem_att_score_cur = att_scores[-1]
                sem_att_score_cur = sem_att_score_cur.squeeze(-1).mean(dim=0)
                sem_att_score_pre = sem_att_score_pre.squeeze(-1).mean(dim=0)
                kd_sem += torch.norm(sem_att_score_cur - sem_att_score_pre, p=2)

            denom = max(1, len(self.teachers) - start_i)
            kd_node /= denom
            kd_sem /= denom
            loss_kd = self.args['lambda_node'] * kd_node + self.args['lambda_sem'] * kd_sem
            meta_loss = self.args['gamma'] * meta_loss + self.args['lambda_er'] * loss_er + \
                        self.args['lambda_kd'] * loss_kd
            
        self.net.zero_grad()
        self.alpha_lr.zero_grad()
        meta_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2)
        torch.nn.utils.clip_grad_norm_(self.alpha_lr.parameters(), 2)
        self.opt.step()
        self.opt_lr.step()
        
        return meta_loss
