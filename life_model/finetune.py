import torch


class NET(torch.nn.Module):
    def __init__(self, model, args):
        super(NET, self).__init__()

        self.net = model
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, g, features):
        logits, all_attention_scores = self.net(g, features)
        return logits, all_attention_scores

    def observe(self, g, features, labels, t, train_mask):
        self.net.train()
        self.net.zero_grad()

        logits, _ = self.net(g, features)
        loss = self.ce(logits[train_mask], labels[train_mask])
        loss.backward()
        self.opt.step()

        return loss