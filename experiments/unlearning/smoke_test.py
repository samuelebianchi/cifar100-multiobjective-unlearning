"""Quick sanity checks (CPU or GPU). Run from repo root:
python -m experiments.unlearning.smoke_test [--epo]
"""
import argparse
import math

import torch
import torch.nn.functional as F

from experiments.unlearning.models import (
    ResNetTarget, ResnetHyper, build_reference_resnet18, chunks_for, get_layer_to_shape,
)
from experiments.unlearning.trainer import forget_loss_fn
from experiments.utils import count_parameters
from phn import EPOSolver, LinearScalarizationSolver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epo", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)

    l2s = get_layer_to_shape(100, cifar_stem=True)
    n = sum(s.numel() for s in l2s.values())
    print(f"[1] target params: {n:,}")

    # functional target == torchvision module (train-mode BN)
    ref = build_reference_resnet18(100, True).train()
    target = ResNetTarget(cifar_stem=True)
    x = torch.randn(16, 3, 32, 32)
    w = {f"resnet.{k}": p.detach() for k, p in ref.named_parameters()}
    with torch.no_grad():
        diff = (ref(x) - target(x, w)).abs().max().item()
    print(f"[2] max |module - functional| = {diff:.2e}")
    assert diff < 1e-3

    # hypernet output shapes
    hnet = ResnetHyper(l2s, num_ws=11, w_dim=10000)
    print(f"[3] HN params {count_parameters(hnet):,}, chunks {hnet.num_chunks} "
          f"(expected {chunks_for(n, 11, 10000)})")
    ray = torch.tensor([0.3, 0.7])
    out = hnet(ray)
    assert list(out) == list(l2s)
    assert all(out[k].shape == l2s[k] for k in l2s)

    # one optimisation step with both objectives
    solver = EPOSolver(2, count_parameters(hnet)) if args.epo else LinearScalarizationSolver(2)
    opt = torch.optim.Adam(hnet.parameters(), lr=1e-4)
    xr, yr = torch.randn(8, 3, 32, 32), torch.randint(1, 100, (8,))
    xf, yf = torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long)
    logits = target(torch.cat([xr, xf]), hnet(ray))
    l_r = F.cross_entropy(logits[:8], yr)
    l_f = forget_loss_fn(logits[8:], yf, "uniform", math.log(100))
    losses = torch.stack((l_r, l_f))
    loss = solver(losses, ray, list(hnet.parameters()))
    loss.backward()
    opt.step()
    print(f"[4] losses retain={l_r.item():.3f} forget={l_f.item():.3f} "
          f"weighted={loss.item():.3f} -> OK")


if __name__ == "__main__":
    main()
