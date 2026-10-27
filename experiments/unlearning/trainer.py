"""Pareto HyperNetwork for class unlearning (CIFAR-100, ResNet18).

Objectives (preference ray = [w_retain, w_forget]):
  f_retain : cross-entropy on a batch of retained-class samples
  f_forget : loss that is >= 0 and small when the forget class is "forgotten"
             ('uniform' = KL(uniform || softmax), 'ce_margin' = relu(margin - CE))

Run from the repo root:  python -m experiments.unlearning.trainer --help
"""
import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from experiments.unlearning.data import UnlearningData, cycle
from experiments.unlearning.models import (
    ResNetTarget, ResnetHyper, build_reference_resnet18, get_layer_to_shape,
)
from experiments.utils import (
    circle_points, count_parameters, get_device, save_args, set_logger, set_seed,
)
from phn import EPOSolver, LinearScalarizationSolver

NUM_CLASSES = 100
HN_NUM_WS = {"11M": 11, "5M": 5, "2M": 2, "1M": 1}


def forget_loss_fn(logits, y, kind, margin):
    if kind == "uniform":  # KL(U || p) >= 0, zero iff the output is uniform
        return -F.log_softmax(logits, dim=1).mean(dim=1).mean() - math.log(logits.size(1))
    ce = F.cross_entropy(logits, y)  # 'ce_margin': push CE above `margin`, then stop
    return F.relu(margin - ce)


def generate_weights(hnet, ray, base):
    w = hnet(ray)
    if base is not None:  # delta mode: theta = theta_pretrained + hnet(ray)
        w = {k: base[k] + v for k, v in w.items()}
    return w


def sample_ray(alpha, device):
    if alpha > 0:
        r = np.random.dirichlet((alpha, alpha)).astype(np.float32)
    else:
        a = np.random.uniform()
        r = np.array([a, 1 - a], dtype=np.float32)
    return torch.from_numpy(r).to(device)


@torch.no_grad()
def evaluate(hnet, net, loader, rays, device, forget_classes, base=None):
    """Per ray: accuracy / CE on retained vs forgotten classes, on MIXED batches."""
    hnet.eval()
    fc = torch.tensor(forget_classes, device=device)
    res = defaultdict(list)
    for ray in rays:
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)
        ray = ray / ray.sum()
        weights = generate_weights(hnet, ray, base)  # one generation per ray
        n = {"r": 0, "f": 0}
        corr = {"r": 0.0, "f": 0.0}
        ce = {"r": 0.0, "f": 0.0}
        for img, y in loader:
            img, y = img.to(device), y.to(device)
            logits = net(img, weights)
            is_f = torch.isin(y, fc)
            ok = logits.argmax(1).eq(y)
            l = F.cross_entropy(logits, y, reduction="none")
            for k, m in (("r", ~is_f), ("f", is_f)):
                n[k] += m.sum().item()
                corr[k] += ok[m].sum().item()
                ce[k] += l[m].sum().item()
        res["ray"].append(ray.cpu().tolist())
        res["retain_acc"].append(corr["r"] / max(n["r"], 1))
        res["forget_acc"].append(corr["f"] / max(n["f"], 1))
        res["retain_ce"].append(ce["r"] / max(n["r"], 1))
        res["forget_ce"].append(ce["f"] / max(n["f"], 1))
    return res


def train(args):
    device = get_device(no_cuda=args.no_cuda, gpus=args.gpus)
    cifar_stem = not args.imagenet_stem

    # ---- nets
    layer_to_shape = get_layer_to_shape(NUM_CLASSES, cifar_stem)
    hnet = ResnetHyper(
        layer_to_shape, hidden_dim=args.ray_hidden, num_ws=HN_NUM_WS[args.hn_size],
        w_dim=args.w_dim, w_init_std=args.w_init_std,
    ).to(device)
    net = ResNetTarget(cifar_stem=cifar_stem).to(device)
    n_target = sum(s.numel() for s in layer_to_shape.values())
    logging.info(f"target params: {n_target:,} | HN params: {count_parameters(hnet):,} "
                 f"| chunks: {hnet.num_chunks}")

    base = None
    if args.base_ckpt:
        ref = build_reference_resnet18(NUM_CLASSES, cifar_stem)
        ref.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"))
        base = {f"resnet.{n}": p.detach().to(device) for n, p in ref.named_parameters()}
        logging.info(f"delta mode: base weights from {args.base_ckpt}")

    # ---- data
    data = UnlearningData(args.data_root, args.forget_classes, args.retain_bs,
                          args.forget_bs, num_workers=args.num_workers)
    logging.info(f"retain: {len(data.retain_set)} | forget: {len(data.forget_set)} "
                 f"(classes {data.forget_classes})")
    forget_iter = cycle(data.forget_loader)

    # ---- optimisation
    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.solver == "epo":
        solver = EPOSolver(n_tasks=2, n_params=count_parameters(hnet))
    else:
        solver = LinearScalarizationSolver(n_tasks=2)

    test_rays = circle_points(args.n_rays, min_angle=0.1, max_angle=np.pi / 2 - 0.1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    pbar = trange(args.n_epochs)
    for epoch in pbar:
        hnet.train()
        sums = np.zeros(2)
        for xr, yr in data.retain_loader:
            xf, yf = next(forget_iter)
            xr, yr, xf, yf = xr.to(device), yr.to(device), xf.to(device), yf.to(device)
            nr = len(xr)

            optimizer.zero_grad()
            ray = sample_ray(args.alpha, device)
            weights = generate_weights(hnet, ray, base)
            # one forward on the concatenated batch -> BN statistics are shared
            # between retain and forget samples (consistent with mixed-batch eval)
            logits = net(torch.cat([xr, xf]), weights)
            l_retain = F.cross_entropy(logits[:nr], yr)
            l_forget = forget_loss_fn(logits[nr:], yf, args.forget_loss, args.forget_margin)
            losses = torch.stack((l_retain, l_forget))

            loss = solver(losses, ray, list(hnet.parameters()))
            loss.backward()
            optimizer.step()
            sums += losses.detach().cpu().numpy()

        sums /= len(data.retain_loader)
        pbar.set_description(f"retain {sums[0]:.3f} | forget {sums[1]:.3f}")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.n_epochs - 1:
            entry = {"test": evaluate(hnet, net, data.test_loader, test_rays, device,
                                      data.forget_classes, base)}
            if not args.no_train_eval:
                entry["train"] = evaluate(hnet, net, data.train_eval_loader, test_rays,
                                          device, data.forget_classes, base)
            results[f"epoch_{epoch + 1}"] = entry
            t = entry["test"]
            logging.info(
                f"epoch {epoch + 1} test | ray0 (retain-heavy): R {t['retain_acc'][-1]:.3f} "
                f"F {t['forget_acc'][-1]:.3f} | ray_last (forget-heavy): "
                f"R {t['retain_acc'][0]:.3f} F {t['forget_acc'][0]:.3f}"
            )
            with open(out_dir / "results.json", "w") as f:
                json.dump(results, f)
            torch.save({"hnet": hnet.state_dict(), "args": vars(args)}, out_dir / "hnet.pt")


def get_args():
    p = argparse.ArgumentParser(description="PHN for CIFAR-100 class unlearning")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--forget-classes", type=int, nargs="+", default=[0])
    p.add_argument("--base-ckpt", type=str, default=None,
                   help="pretrained ResNet18 state_dict -> delta mode (recommended)")
    p.add_argument("--solver", type=str, choices=["ls", "epo"], default="ls")
    p.add_argument("--forget-loss", type=str, choices=["uniform", "ce_margin"], default="uniform")
    p.add_argument("--forget-margin", type=float, default=math.log(NUM_CLASSES))
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.2, help="Dirichlet alpha for rays")
    p.add_argument("--hn-size", type=str, choices=list(HN_NUM_WS), default="11M")
    p.add_argument("--w-dim", type=int, default=10000)
    p.add_argument("--w-init-std", type=float, default=0.1,
                   help="std of W init; use ~1e-3 in delta mode")
    p.add_argument("--ray-hidden", type=int, default=100)
    p.add_argument("--imagenet-stem", action="store_true",
                   help="7x7/stride-2 stem + maxpool instead of the CIFAR stem")
    p.add_argument("--retain-bs", type=int, default=128)
    p.add_argument("--forget-bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--n-rays", type=int, default=11)
    p.add_argument("--no-train-eval", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--out-dir", type=str, default="outputs/unlearning")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    set_logger()
    save_args(folder=args.out_dir, args=args)
    train(args)
