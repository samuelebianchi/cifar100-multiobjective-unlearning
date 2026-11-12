"""CIFAR-100 splits for class unlearning: retain set / forget set / eval sets."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from experiments.utils import seed_worker

logger = logging.getLogger(__name__)

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build the CIFAR-100 transforms.

    Returns:
        A pair ``(train_tf, eval_tf)``: random crop + horizontal flip + normalization for
        training, normalization only for evaluation.
    """
    norm = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), norm])
    return train_tf, eval_tf


def cycle(loader: DataLoader) -> Iterator[Any]:
    """Iterate endlessly over a DataLoader (the forget set is tiny).

    Args:
        loader: The data loader to repeat.

    Yields:
        Batches of ``loader``, restarting from the first one when it is exhausted.
    """
    while True:
        yield from loader


class UnlearningData:
    """CIFAR-100 data for class unlearning.

    The forget set contains all the training samples of the classes to forget. By default the
    retain set is a random subset, of the same size, of the training samples of the other
    classes (``retain_ratio=1``); with ``retain_ratio=r`` it has ``r`` times as many samples as
    the forget set, and with ``retain_ratio=None`` it contains all of them.

    Attributes:
        forget_classes: Sorted list of the classes to forget.
        retain_set: Training samples of the retained classes used for training (with
            augmentation): a random subset as large as the forget set, or all of them.
        forget_set: All the training samples of the classes to forget (with augmentation).
        train_eval: Full training set without augmentation (for evaluation).
        test: Full test set.
        retain_loader: Shuffled loader over ``retain_set``.
        forget_loader: Shuffled loader over ``forget_set``.
        train_eval_loader: Sequential loader over ``train_eval``.
        test_loader: Sequential loader over ``test``.
    """

    def __init__(
        self,
        root: str | Path,
        forget_classes: Sequence[int],
        retain_bs: int = 128,
        forget_bs: int = 32,
        eval_bs: int = 500,
        num_workers: int = 4,
        download: bool = True,
        retain_ratio: float | None = 1.0,
        seed: int = 42,
    ) -> None:
        """Build the datasets and the data loaders.

        Args:
            root: Directory where CIFAR-100 is stored (or downloaded).
            forget_classes: Labels of the classes to forget.
            retain_bs: Batch size of the retain loader.
            forget_bs: Batch size of the forget loader (capped to the size of the forget set).
            eval_bs: Batch size of the evaluation loaders.
            num_workers: Number of worker processes of each loader.
            download: Whether to download CIFAR-100 if it is missing.
            retain_ratio: Size of the retain set relative to the forget set: the retain set is
                a random subset of the training samples of the retained classes with
                ``round(retain_ratio * len(forget_set))`` samples (capped to the number of
                available samples). 1 = same size, 2 = twice as large, and so on. If None
                the retain set contains all the samples of the retained classes.
            seed: Seed of the retain subset selection and of the shuffling of the loaders.
        """
        train_tf, eval_tf = get_transforms()
        train_aug = datasets.CIFAR100(root, train=True, download=download, transform=train_tf)
        self.train_eval = datasets.CIFAR100(root, train=True, download=False, transform=eval_tf)
        self.test = datasets.CIFAR100(root, train=False, download=download, transform=eval_tf)

        self.forget_classes = sorted({int(c) for c in forget_classes})
        targets = np.asarray(train_aug.targets)
        is_forget = np.isin(targets, self.forget_classes)
        forget_idx = np.flatnonzero(is_forget)
        retain_idx = np.flatnonzero(~is_forget)
        if retain_ratio is not None:
            if retain_ratio <= 0:
                raise ValueError(f"retain_ratio must be > 0, got {retain_ratio}")
            n_retain = min(round(retain_ratio * len(forget_idx)), len(retain_idx))
            if n_retain < round(retain_ratio * len(forget_idx)):
                logger.warning(
                    f"only {len(retain_idx)} retain samples available, "
                    f"{round(retain_ratio * len(forget_idx))} requested "
                    f"(ratio {retain_ratio} x {len(forget_idx)} forget samples): "
                    "the retain set is capped"
                )
            rng = np.random.default_rng(seed)
            retain_idx = np.sort(rng.choice(retain_idx, size=n_retain, replace=False))
        self.retain_set = Subset(train_aug, retain_idx)
        self.forget_set = Subset(train_aug, forget_idx)

        # one seeded generator per shuffled loader: the order of the batches is reproducible
        g_retain, g_forget = torch.Generator(), torch.Generator()
        g_retain.manual_seed(seed)
        g_forget.manual_seed(seed + 1)
        kw = {"num_workers": num_workers, "pin_memory": True, "worker_init_fn": seed_worker}
        self.retain_loader = DataLoader(
            self.retain_set,
            batch_size=retain_bs,
            shuffle=True,
            drop_last=True,
            generator=g_retain,
            **kw,
        )
        fbs = min(forget_bs, len(self.forget_set))
        self.forget_loader = DataLoader(
            self.forget_set,
            batch_size=fbs,
            shuffle=True,
            drop_last=True,
            generator=g_forget,
            **kw,
        )
        self.train_eval_loader = DataLoader(
            self.train_eval, batch_size=eval_bs, shuffle=False, **kw
        )
        self.test_loader = DataLoader(self.test, batch_size=eval_bs, shuffle=False, **kw)