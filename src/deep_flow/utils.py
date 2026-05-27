"""Shared training and evaluation utilities."""

from pathlib import Path
import json
import math
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def compute_lr(epoch, epochs, min_lr, max_lr):
    """Original second-half exponential learning-rate decay schedule."""
    if epoch < epochs * 0.5:
        return max_lr
    exponent = (epoch / float(epochs) - 0.5) * 2.0
    exponent = exponent * 6.0
    factor = math.pow(0.5, exponent)
    return min_lr + (max_lr - min_lr) * factor


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def average(self):
        return self.total / max(self.count, 1)
