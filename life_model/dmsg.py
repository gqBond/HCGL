import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.data_loader import get_label_ids
from .dmsg_utils import my_sampler
from .hero_utils import sub_g_sampling


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, rate):
        ctx.rate = rate
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.rate * gradient, None


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs):
        return self.layers(GradientReverse.apply(inputs, 1.0))


class NET(nn.Module):
    """DMSG adapted to the heterogeneous HAN backbone."""

    def __init__(self, model, hg, args):
        super().__init__()
        self.net = model
        self.args = args
        self.dataset = args["dataset"]
        self.n_tasks = args["num_tasks"]
        self.get_subgraph = sub_g_sampling(self.dataset)

        dataset_classes = {"DBLP": 4, "Freebase": 7, "IMDB": 5}
        self.n_classes = dataset_classes[self.dataset]
        self.n_cls_per_task = self.n_classes // self.n_tasks

        self.dmsg_args = args["dmsg_args"]
        sampler_name = self.dmsg_args.get("sampler", "My")
        if sampler_name != "My":
            raise ValueError(f"Unknown DMSG sampler: {sampler_name}")
        self.sampler = my_sampler(args)
        self.budget = int(self.dmsg_args.get("budget", 50))
        self.loss_weights = self.dmsg_args.get(
            "loss_weights", [1.0, 10.0, 1.0, 1.0]
        )
        if len(self.loss_weights) != 4:
            raise ValueError("DMSG loss_weights must contain four values")

        label_order = get_label_ids(self.dataset)
        self.local_label = {
            global_label: position % self.n_cls_per_task
            for position, global_label in enumerate(label_order)
        }

        latent_size = self.net.predict.in_features
        self.mean_layer = nn.Linear(latent_size, latent_size)
        self.scale_layer = nn.Linear(latent_size, latent_size)
        self.discriminator = Discriminator(latent_size)

        self.ce = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(
            self.parameters(),
            lr=args["lr"],
            weight_decay=args["weight_decay"],
        )

        self.current_task = -1
        self.buffer_node_ids = []
        self.last_graph = None
        self.last_features = None
        self.last_ids_per_class = None

    def forward(self, graph, features):
        output, _ = self.net(graph, features)
        return output

    def _variational_forward(self, graph, features):
        _, attention = self.net(graph, features)
        hidden = self.net.last_h
        mean = self.mean_layer(hidden)
        scale = F.softplus(self.scale_layer(hidden), beta=10) + 1e-8
        sampled_hidden = mean + torch.randn_like(scale) * scale
        logits = self.net.predict(sampled_hidden)
        return logits, attention, sampled_hidden, mean, scale

    def _store_previous_task(self):
        if self.last_graph is None:
            return

        ordered_ids = [
            int(node_id)
            for class_ids in self.last_ids_per_class
            for node_id in class_ids
        ]
        if not ordered_ids:
            return

        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(self.last_graph, self.last_features)
            representations = logits[ordered_ids]
            sampled_ids = self.sampler(
                self.last_ids_per_class,
                self.budget,
                representations,
            )

        original_ids = self.last_graph.nodes["0"].data["_ID"]
        sampled_ids = torch.tensor(
            sampled_ids,
            dtype=torch.long,
            device=original_ids.device,
        )
        self.buffer_node_ids.extend(original_ids[sampled_ids].cpu().tolist())

    def observe(
        self,
        graph,
        full_graph,
        full_features,
        features,
        full_labels,
        labels,
        task_id,
        train_ids,
        ids_per_cls_train,
    ):
        if task_id != self.current_task:
            if self.current_task >= 0:
                self._store_previous_task()
            self.current_task = task_id
            self.last_graph = graph
            self.last_features = features
            self.last_ids_per_class = ids_per_cls_train

        self.train()
        self.opt.zero_grad()

        logits, _ = self.net(graph, features)
        task_loss = self.ce(logits[train_ids], labels[train_ids])
        loss = task_loss

        if self.buffer_node_ids:
            replay_graph = self.get_subgraph(
                full_graph,
                self.buffer_node_ids,
                full_features,
                full_labels,
                self.args["budget_ots"],
            ).to(self.args["device"])
            replay_features = replay_graph.ndata["features"]["0"]
            replay_labels = replay_graph.ndata["labels"]["0"].squeeze()
            if self.dataset == "Freebase":
                replay_features = replay_features.to_sparse()
            replay_features = replay_features.to(self.args["device"])
            replay_labels = torch.tensor(
                [self.local_label[int(label)] for label in replay_labels.cpu().tolist()],
                dtype=torch.long,
                device=self.args["device"],
            )

            replay_logits, _, hidden, mean, scale = self._variational_forward(
                replay_graph, replay_features
            )
            replay_loss = self.ce(replay_logits, replay_labels)

            similarity = hidden @ hidden.t()
            label_similarity = (
                replay_labels.unsqueeze(1) == replay_labels.unsqueeze(0)
            ).float()
            off_diagonal = 1 - torch.eye(
                similarity.shape[0], device=similarity.device
            )
            variational_loss = F.binary_cross_entropy_with_logits(
                similarity,
                label_similarity,
                weight=off_diagonal,
            )
            variational_loss += -0.5 * (
                1 + 2 * scale.log() - mean.pow(2) - scale.pow(2)
            ).sum(1).mean() / math.log(2)

            discriminator_input = torch.cat((hidden, mean), dim=0)
            discriminator_logits = self.discriminator(discriminator_input).squeeze(-1)
            discriminator_labels = torch.cat(
                (
                    torch.ones(hidden.shape[0], device=hidden.device),
                    torch.zeros(mean.shape[0], device=mean.device),
                ),
                dim=0,
            )
            adversarial_loss = F.binary_cross_entropy_with_logits(
                discriminator_logits, discriminator_labels
            )

            loss = (
                self.loss_weights[0] * task_loss
                + self.loss_weights[1] * replay_loss
                + self.loss_weights[2] * adversarial_loss
                + self.loss_weights[3] * variational_loss
            )

        loss.backward()
        self.opt.step()
        return loss
