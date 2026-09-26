"""Small utilities shared by the training entry point."""

import random
from pathlib import Path

import dgl
import numpy as np
import torch


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected a Boolean value, received {value!r}")


def str2dict(value):
    """Parse ``key:value;key:[value,value]`` command-line dictionaries."""
    output = {}
    for pair in value.replace(" ", "").replace("'", "").split(";"):
        key, raw_value = pair.split(":", maxsplit=1)
        values = raw_value.strip("[]").split(",")
        parsed = []
        for item in values:
            try:
                parsed.append(float(item))
            except ValueError:
                parsed.append(item)
        output[key] = parsed if raw_value.startswith("[") else parsed[0]
    return output


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience, checkpoint_path):
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif loss > self.best_loss and acc <= self.best_acc:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if loss <= self.best_loss and acc >= self.best_acc:
                self.save_checkpoint(model)
            self.best_loss = min(loss, self.best_loss)
            self.best_acc = max(acc, self.best_acc)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self, model, map_location=None):
        state = torch.load(self.checkpoint_path, map_location=map_location)
        model.load_state_dict(state)
