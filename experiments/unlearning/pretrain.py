"""Standard CIFAR-100 training of the reference ResNet18 (the model to unlearn from).

python -m experiments.unlearning.pretrain --out checkpoints/resnet18_cifar100.pt
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from experiments.unlearning.data import get_transforms
from experiments.unlearning.models import build_reference_resnet18
from experiments.utils import get_device, set_logger, set_seed


@torch.no_grad()
def test_acc(model, loader, device):
    model.eval()
    ok = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ok += model(x).argmax(1).eq(y).sum().item()
        n += len(y)
    return ok / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="checkpoints/resnet18_cifar100.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpus", default="0")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    set_seed(a.seed)
    set_logger()
    device = get_device(gpus=a.gpus)

    train_tf, eval_tf = get_transforms()
    train_set = datasets.CIFAR100(a.data_root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR100(a.data_root, train=False, download=True, transform=eval_tf)
    kw = dict(num_workers=a.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, a.bs, shuffle=True, drop_last=True, **kw)
    test_loader = DataLoader(test_set, 500, shuffle=False, **kw)

    model = build_reference_resnet18(100, cifar_stem=True).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9,
                          weight_decay=a.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    for ep in range(a.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()
        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep + 1}/{a.epochs} test acc {test_acc(model, test_loader, device):.4f}")

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
