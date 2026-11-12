"""Pareto HyperNetwork for class unlearning (CIFAR-100, ResNet18).

Objectives (preference ray = [w_retain, w_forget]):
  f_retain : cross-entropy ('ce') or KL from the original model ('kl') on retain samples
  f_forget : 'uniform' = KL(U_S || softmax), >= 0, with U_S the
                         uniform distribution over the surviving classes only
             'neg_ce'  = -CE, i.e. gradient ascent: unbounded, negative, NOT usable with EPO;
                         with --forget-ce-cap c it becomes -min(CE, c): same gradient as the
                         ascent until CE reaches c, zero beyond, so the objective is bounded

Run from the repo root:  python -m experiments.unlearning.trainer --help
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from experiments.unlearning.data import UnlearningData, cycle
from experiments.unlearning.models import (
    ResnetHyper,
    build_reference_resnet18,
    forward_with_weights,
    get_layer_to_shape,
)
from experiments.utils import (
    circle_points,
    count_parameters,
    get_device,
    save_args,
    set_logger,
    set_seed,
)
from phn import EPOSolver, LinearScalarizationSolver

NUM_CLASSES = 100
HN_NUM_WS = {"11M": 11, "5M": 5, "2M": 2, "1M": 1}

logger = logging.getLogger(__name__)


def retain_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    kind: str,
    base_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample retain objective, to be minimised.

    Args:
        logits: Model outputs, shape ``(B, C)``.
        y: True labels, shape ``(B,)`` (used only by ``ce``).
        kind: ``"kl"`` for KL(p_theta0 || p_theta), ``"ce"`` for the cross-entropy.
        base_logits: Outputs of the original model theta_0 on the same samples (``kl`` only).

    Returns:
        Loss of each sample, shape ``(B,)``; the batch loss is its mean.
    """
    if kind not in ("kl", "ce"):
        raise ValueError(f"unknown retain loss kind: {kind!r}")
    if kind == "kl":  # KL(p_theta0 || p_theta) on retain samples
        return F.kl_div(
            F.log_softmax(logits, dim=1), F.softmax(base_logits, dim=1), reduction="none"
        ).sum(dim=1)
    return F.cross_entropy(logits, y, reduction="none")


def forget_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    kind: str,
    forget_classes: torch.Tensor,
    ce_cap: float | None = None,
) -> torch.Tensor:
    """Per-sample forget objective, to be minimised.

    Args:
        logits: Model outputs, shape ``(B, C)``.
        y: True labels, shape ``(B,)`` (used only by ``neg_ce``).
        kind: ``"uniform"`` for KL(U_S || p_theta), ``"neg_ce"`` for ``-CE``. U_S is the
            uniform distribution over the surviving classes S (all the classes except the
            forgotten ones) and p_theta the softmax over all the classes.
        forget_classes: Labels of the forgotten classes, shape ``(k,)`` (``uniform`` only).
        ce_cap: Only for ``neg_ce``: the cross-entropy is clamped at this value, so the loss is
            ``-min(CE, ce_cap)``. If None the ascent is unbounded.

    Returns:
        Loss of each sample, shape ``(B,)``; the batch loss is its mean. ``neg_ce`` is negative
        and unbounded below unless ``ce_cap`` is set.
    """
    if kind == "uniform":
        # U_S: uniform over the surviving classes, 0 on the forgotten ones
        target = torch.ones(logits.size(1), device=logits.device)
        target[forget_classes] = 0.0
        target = (target / target.sum()).expand_as(logits)
        return F.kl_div(F.log_softmax(logits, dim=1), target, reduction="none").sum(dim=1)
    if kind != "neg_ce":
        raise ValueError(f"unknown forget loss kind: {kind!r}")
    # 'neg_ce': gradient ascent on the forget set. The solver minimizes the loss, so the sign
    # must be negative.
    ce = F.cross_entropy(logits, y, reduction="none")
    if ce_cap is not None:  # beyond the cap the gradient is zero, so the objective is bounded
        ce = ce.clamp(max=ce_cap)
    return -ce


def log_eval(tag: str, split: str, res: dict[str, list]) -> None:
    """Log an evaluation in two readable lines: the first ray (lowest w_forget), the last one (highest w_forget).

    'loss' is the training objective of each subset (retain loss and forget loss), evaluated on
    the samples of the retained classes or of the forgotten classses.

    Args:
        tag: Prefix of the log lines (e.g. the epoch).
        split: Name of the evaluated split (``"test"`` or ``"train"``).
        res: Result dictionary returned by `evaluate`.
    """
    for i, name in ((0, "ray favouring RETAIN"), (-1, "ray favouring FORGET")):
        w = res["ray"][i]
        logger.info(
            f"{tag} | {split} set | {name} (weight retain={w[0]:.2f}, forget={w[1]:.2f}) | "
            f"retained classes: accuracy {res['retain_acc'][i]:.3f}, "
            f"loss {res['retain_loss'][i]:.4f} | "
            f"forgotten class: accuracy {res['forget_acc'][i]:.3f}, "
            f"loss {res['forget_loss'][i]:.4f}"
        )


def generate_weights(
    hnet: ResnetHyper, ray: torch.Tensor, base: dict[str, torch.Tensor] | None
) -> dict[str, torch.Tensor]:
    """Generate the weights of the target network for a preference ray.

    Args:
        hnet: The hypernetwork.
        ray: Preference ray ``[w_retain, w_forget]``.
        base: Weights of the pretrained model; if given the hypernetwork output is added to them.

    Returns:
        Dictionary mapping each parameter name to its tensor.
    """
    w = hnet(ray)
    if base is not None:  # theta = theta_pretrained + hnet(ray)
        w = {k: base[k] + v for k, v in w.items()}
    return w


def sample_ray(alpha: float, device: torch.device) -> torch.Tensor:
    """Sample a random preference ray.

    Args:
        alpha: Concentration of the Dirichlet distribution; if <= 0 the first weight is
            sampled uniformly in [0, 1].
        device: Device of the returned tensor.

    Returns:
        Preference ray of shape ``(2,)`` that sums to one.
    """
    if alpha > 0:
        r = np.random.dirichlet((alpha, alpha)).astype(np.float32)
    else:
        a = np.random.uniform()
        r = np.array([a, 1 - a], dtype=np.float32)
    return torch.from_numpy(r).to(device)


@torch.no_grad()
def evaluate(
    hnet: ResnetHyper,
    net: nn.Module,
    loader: DataLoader,
    rays: np.ndarray,
    device: torch.device,
    forget_classes: Sequence[int],
    base: dict[str, torch.Tensor] | None = None,
    loss_kinds: tuple[str, str] = ("ce", "uniform"),
    forget_ce_cap: float | None = None,
) -> dict[str, list]:
    """Evaluate the hypernetwork on a set of rays, using MIXED batches.

    For each ray it computes the accuracy and the training objective, separately for the samples
    of the retained classes and for those of the forgotten classes. The objective is the one
    used in training: the retain loss (``--retain-loss``) on the retained-class samples and the
    forget loss (``--forget-loss``) on the forgotten-class samples, averaged over the samples.

    Args:
        hnet: The hypernetwork.
        net: The target network (torchvision ResNet18), called with the generated weights.
        loader: Sequential loader over the evaluation set.
        rays: Preference rays ``[w_retain, w_forget]``, shape ``(n_rays, 2)``.
        device: Device where the evaluation runs.
        forget_classes: Labels of the forgotten classes.
        base: Weights of the pretrained model or None.
        loss_kinds: ``(retain_kind, forget_kind)`` of the training objectives.
        forget_ce_cap: Cap of the cross-entropy of the ``neg_ce`` forget objective (or None).

    Returns:
        Dictionary of lists (one entry per ray) with keys ``ray``, ``retain_acc``,
        ``forget_acc``, ``retain_loss`` and ``forget_loss``.
    """
    retain_kind, forget_kind = loss_kinds
    hnet.eval()
    forget_labels = torch.tensor(forget_classes, device=device)
    results = defaultdict(list)
    base_logits_cache = []  # theta_0 logits of each batch: same for every ray, computed once
    for ray in rays:
        ray_tensor = torch.from_numpy(ray.astype(np.float32)).to(device)
        ray_tensor = ray_tensor / ray_tensor.sum()
        weights = generate_weights(hnet, ray_tensor, base)

        subsets = ("retain", "forget")
        num_samples = dict.fromkeys(subsets, 0)
        num_correct = dict.fromkeys(subsets, 0)
        loss_sum = dict.fromkeys(subsets, 0.0)
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            logits = forward_with_weights(net, images, weights)
            base_logits = None
            if retain_kind == "kl":
                if len(base_logits_cache) <= batch_idx:
                    base_logits_cache.append(forward_with_weights(net, images, base))
                base_logits = base_logits_cache[batch_idx]
            is_forgotten = torch.isin(labels, forget_labels)
            is_correct = logits.argmax(dim=1).eq(labels)
            loss_per_sample = {
                "retain": retain_loss(logits, labels, retain_kind, base_logits),
                "forget": forget_loss(logits, labels, forget_kind, forget_labels, forget_ce_cap),
            }
            for subset, in_subset in (("retain", ~is_forgotten), ("forget", is_forgotten)):
                num_samples[subset] += in_subset.sum().item()
                num_correct[subset] += is_correct[in_subset].sum().item()
                loss_sum[subset] += loss_per_sample[subset][in_subset].sum().item()

        results["ray"].append(ray_tensor.cpu().tolist())
        for subset in subsets:
            n = max(num_samples[subset], 1)
            results[f"{subset}_acc"].append(num_correct[subset] / n)
            results[f"{subset}_loss"].append(loss_sum[subset] / n)
    return results


def train(args: argparse.Namespace) -> None:
    """Train the hypernetwork and periodically evaluate it.

    Results are written to ``<out_dir>/results.json``, the training objectives of every epoch to
    ``<out_dir>/train_log.json`` and the hypernetwork to ``<out_dir>/hnet.pt`` after each
    evaluation.

    Args:
        args: Parsed command-line arguments, see `get_args`.

    Raises:
        ValueError: If ``--retain-loss kl`` is used without ``--base-ckpt``.
    """
    if args.retain_loss == "kl" and not args.base_ckpt:
        raise ValueError("--retain-loss kl needs --base-ckpt")
    if args.forget_ce_cap is not None and args.forget_loss != "neg_ce":
        raise ValueError("--forget-ce-cap only applies to --forget-loss neg_ce")
    if args.full_retain and args.retain_ratio is not None:
        raise ValueError("--full-retain and --retain-ratio are mutually exclusive")
    if args.retain_ratio is not None and args.retain_ratio <= 0:
        raise ValueError("--retain-ratio must be > 0")
    device = get_device(no_cuda=args.no_cuda, gpus=args.gpus)

    # ---- nets
    layer_to_shape = get_layer_to_shape(NUM_CLASSES)
    hnet = ResnetHyper(
        layer_to_shape,
        hidden_dim=args.ray_hidden,
        num_ws=HN_NUM_WS[args.hn_size],
        w_dim=args.w_dim,
        w_init_std=args.w_init_std,
    ).to(device)
    # target: the torchvision ResNet18 itself. Its own parameters are never used (the generated
    # weights replace them at every call, see `forward_with_weights` function), only its buffers are.
    net = build_reference_resnet18(NUM_CLASSES).to(device)
    net.requires_grad_(False)
    n_target = sum(s.numel() for s in layer_to_shape.values())
    logger.info(
        f"target params: {n_target:,} | HN params: {count_parameters(hnet):,} "
        f"| chunks: {hnet.num_chunks}"
    )

    base = None
    if args.base_ckpt:
        net.load_state_dict(torch.load(args.base_ckpt, map_location=device, weights_only=True))
        base = {n: p.detach().clone() for n, p in net.named_parameters()}
        # eval mode: BatchNorm uses the frozen statistics of the pretrained model, both in
        # training and in evaluation
        net.eval()
        logger.info(f"delta mode: base weights + frozen BN stats from {args.base_ckpt}")
    else:
        # BatchNorm uses batch statistics
        net.train()

    # ---- data
    data = UnlearningData(
        args.data_root,
        args.forget_classes,
        args.retain_bs,
        args.forget_bs,
        num_workers=args.num_workers,
        retain_ratio=None if args.full_retain else (args.retain_ratio or 1.0),
        seed=args.seed,
    )
    logger.info(
        f"retain: {len(data.retain_set)} | forget: {len(data.forget_set)} "
        f"(classes {data.forget_classes})"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "data_info.json").open("w") as f:
        json.dump(
            {
                "forget_classes": data.forget_classes,
                "retain_size": len(data.retain_set),
                "forget_size": len(data.forget_set),
                "effective_retain_ratio": len(data.retain_set) / len(data.forget_set),
            },
            f,
        )
    forget_iter = cycle(data.forget_loader)
    forget_classes = torch.tensor(data.forget_classes, device=device)

    # ---- optimization
    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.solver == "epo":
        solver = EPOSolver(n_tasks=2, n_params=count_parameters(hnet))
    else:
        solver = LinearScalarizationSolver(n_tasks=2)

    # evaluation rays [w_retain, w_forget]
    test_rays = np.vstack(
        [
            [1.0, 0.0],
            circle_points(args.n_rays, min_angle=0.1, max_angle=np.pi / 2 - 0.1),
            [0.0, 1.0],
        ]
    )
    results = {}

    # evaluation of the starting point (before any training), same pipeline as later epochs
    loss_kinds = (args.retain_loss, args.forget_loss)
    t0 = evaluate(
        hnet,
        net,
        data.test_loader,
        test_rays,
        device,
        data.forget_classes,
        base,
        loss_kinds,
        args.forget_ce_cap,
    )
    results["epoch_0"] = {"test": t0}
    log_eval("epoch 0 (initial model)", "test", t0)

    # mean training objectives of each epoch (over the steps, each with a different random ray)
    train_log = {"retain_loss": [], "forget_loss": []}
    pbar = trange(args.n_epochs)
    for epoch in pbar:
        hnet.train()
        objective_sums = np.zeros(2)
        for xr, yr in data.retain_loader:
            xf, yf = next(forget_iter)
            xr, yr, xf, yf = xr.to(device), yr.to(device), xf.to(device), yf.to(device)
            nr = len(xr)

            optimizer.zero_grad()
            ray = sample_ray(args.alpha, device)
            # ray = torch.tensor([0.5, 0.5], device=device)
            weights = generate_weights(hnet, ray, base)
            # single forward on the concatenated batch
            logits = forward_with_weights(net, torch.cat([xr, xf]), weights)
            base_logits = None
            if args.retain_loss == "kl":
                with torch.no_grad(): 
                    base_logits = forward_with_weights(net, xr, base)
            l_retain = retain_loss(logits[:nr], yr, args.retain_loss, base_logits).mean()
            l_forget = forget_loss(
                logits[nr:], yf, args.forget_loss, forget_classes, args.forget_ce_cap
            ).mean()
            losses = torch.stack((l_retain, l_forget))

            loss = solver(losses, ray, list(hnet.parameters()))
            loss.backward()
            optimizer.step()
            objective_sums += losses.detach().cpu().numpy()

        # mean over the steps of the epoch (one step per batch of the retain loader)
        objective_sums /= len(data.retain_loader)
        train_log["retain_loss"].append(float(objective_sums[0]))
        train_log["forget_loss"].append(float(objective_sums[1]))
        pbar.set_description(
            f"train loss | retain set {objective_sums[0]:.4f} | forget set {objective_sums[1]:.4f}"
        )

        if (epoch + 1) % args.eval_every == 0 or epoch == args.n_epochs - 1:
            entry = {
                "test": evaluate(
                    hnet,
                    net,
                    data.test_loader,
                    test_rays,
                    device,
                    data.forget_classes,
                    base,
                    loss_kinds,
                    args.forget_ce_cap,
                )
            }
            if not args.no_train_eval:
                entry["train"] = evaluate(
                    hnet,
                    net,
                    data.train_eval_loader,
                    test_rays,
                    device,
                    data.forget_classes,
                    base,
                    loss_kinds,
                    args.forget_ce_cap,
                )
            results[f"epoch_{epoch + 1}"] = entry
            for split in ("test", "train"):
                if split in entry:
                    log_eval(f"epoch {epoch + 1}", split, entry[split])
            with (out_dir / "results.json").open("w") as f:
                json.dump(results, f)
            with (out_dir / "train_log.json").open("w") as f:
                json.dump(train_log, f)
            torch.save({"hnet": hnet.state_dict(), "args": vars(args)}, out_dir / "hnet.pt")


def get_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="PHN for CIFAR-100 class unlearning")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--forget-classes", type=int, nargs="+", default=[0])
    p.add_argument(
        "--base-ckpt",
        type=str,
        default=None,
        help="pretrained ResNet18 state_dict -> delta mode (recommended)",
    )
    p.add_argument("--solver", type=str, choices=["ls", "epo"], default="ls")
    p.add_argument("--forget-loss", type=str, choices=["uniform", "neg_ce"], default="neg_ce")
    p.add_argument(
        "--forget-ce-cap",
        type=float,
        default=None,
        help="neg_ce only: forget loss = -min(CE, cap) instead of the unbounded -CE",
    )
    p.add_argument(
        "--retain-loss",
        type=str,
        choices=["ce", "kl"],
        default="ce",
        help="ce: cross-entropy on labels; kl: KL(p_theta0 || p_theta), needs --base-ckpt",
    )
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.2, help="Dirichlet alpha for rays")
    p.add_argument("--hn-size", type=str, choices=list(HN_NUM_WS), default="11M")
    p.add_argument("--w-dim", type=int, default=10000)
    p.add_argument(
        "--w-init-std",
        type=float,
        default=0.1,
        help="std of W init; use ~1e-3 in delta mode",
    )
    p.add_argument("--ray-hidden", type=int, default=100)
    p.add_argument(
        "--full-retain",
        action="store_true",
        help="use all the samples of the retained classes as retain set "
        "(default: random subset as large as the forget set; see --retain-ratio)",
    )
    p.add_argument(
        "--retain-ratio",
        type=float,
        default=None,
        help="retain set size / forget set size (1 = same size, 2 = twice, ...); "
        "default 1; incompatible with --full-retain",
    )
    p.add_argument("--retain-bs", type=int, default=128)
    p.add_argument("--forget-bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument(
        "--n-rays",
        type=int,
        default=11,
        help="interior evaluation rays; the extremes (1,0) and (0,1) are always added",
    )
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