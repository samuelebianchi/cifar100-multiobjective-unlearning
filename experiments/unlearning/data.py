"""CIFAR-100 splits for class unlearning: retain set / forget set / eval sets."""
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


def get_transforms():
    norm = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), norm]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), norm])
    return train_tf, eval_tf


def cycle(loader):
    """Endless iterator over a DataLoader (the forget set is tiny)."""
    while True:
        for batch in loader:
            yield batch


class UnlearningData:
    def __init__(self, root, forget_classes, retain_bs=128, forget_bs=32,
                 eval_bs=500, num_workers=4, download=True):
        train_tf, eval_tf = get_transforms()
        train_aug = datasets.CIFAR100(root, train=True, download=download, transform=train_tf)
        self.train_eval = datasets.CIFAR100(root, train=True, download=False, transform=eval_tf)
        self.test = datasets.CIFAR100(root, train=False, download=download, transform=eval_tf)

        self.forget_classes = sorted({int(c) for c in forget_classes})
        targets = np.asarray(train_aug.targets)
        is_forget = np.isin(targets, self.forget_classes)
        self.retain_set = Subset(train_aug, np.flatnonzero(~is_forget))
        self.forget_set = Subset(train_aug, np.flatnonzero(is_forget))

        kw = dict(num_workers=num_workers, pin_memory=True)
        self.retain_loader = DataLoader(self.retain_set, batch_size=retain_bs,
                                        shuffle=True, drop_last=True, **kw)
        fbs = min(forget_bs, len(self.forget_set))
        self.forget_loader = DataLoader(self.forget_set, batch_size=fbs,
                                        shuffle=True, drop_last=True, **kw)
        self.train_eval_loader = DataLoader(self.train_eval, batch_size=eval_bs,
                                            shuffle=False, **kw)
        self.test_loader = DataLoader(self.test, batch_size=eval_bs, shuffle=False, **kw)
