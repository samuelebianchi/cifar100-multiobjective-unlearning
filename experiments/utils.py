import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed: seed of all the random number generators
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """``worker_init_fn`` of a DataLoader: seeds numpy and random in every worker.

    Use it together with a seeded ``torch.Generator`` passed to the DataLoader, as in the
    PyTorch reproducibility notes.
    :param worker_id: index of the worker (unused, the seed is derived from torch)
    :return:
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def save_args(folder, args, name="config.json", check_exists=False):
    set_logger()
    path = Path(folder)
    if check_exists:
        if path.exists():
            logging.warning(f"folder {folder} already exists! old files might be lost.")
    path.mkdir(parents=True, exist_ok=True)

    json.dump(vars(args), open(path / name, "w"))


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]
