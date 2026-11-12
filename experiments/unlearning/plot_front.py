"""Table + plots of the retain/forget trade-off from the results of a run.

python -m experiments.unlearning.plot_front outputs/ls_c0 [--epoch 50] [--split test|train]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RETAIN_LOSS_NAMES = {
    "ce": "cross-entropy",
    "kl": "KL(p_θ0 || p_θ)",
}
FORGET_LOSS_NAMES = {
    "uniform": "KL(U_surviving || p_θ)",
    "neg_ce": "-cross-entropy",
}


def describe_forget_classes(classes: list[int], max_shown: int = 6) -> str:
    """Short description of the forgotten classes (e.g. ``"0"`` or ``"0-19 (20 classes)"``).

    Args:
        classes: Labels of the forgotten classes.
        max_shown: Above this number the classes are summarised as a range.

    Returns:
        A short human-readable string.
    """
    if len(classes) <= max_shown:
        return ",".join(map(str, classes))
    if classes == list(range(classes[0], classes[-1] + 1)):
        return f"{classes[0]}-{classes[-1]} ({len(classes)} classes)"
    return f"{len(classes)} classes"


def main() -> None:
    """Print the trade-off table of a run and save the plots next to its results.

    Reads ``<run_dir>/results.json``, ``<run_dir>/config.json`` and
    ``<run_dir>/train_log.json`` and writes ``front_<split>_epoch<N>.png`` in ``<run_dir>``.
    The plots are: the front in loss space, the accuracies along the rays, the retain accuracy
    at the evaluated epochs and the training objectives of every epoch.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--epoch", type=int, default=None, help="default: last evaluated epoch")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    args = ap.parse_args()

    run = Path(args.run_dir)
    with (run / "results.json").open() as f:
        res = json.load(f)
    with (run / "config.json").open() as f:
        cfg = json.load(f)
    with (run / "train_log.json").open() as f:
        train_log = json.load(f)
    epochs = sorted(int(k.split("_")[1]) for k in res)
    ep = args.epoch if args.epoch is not None else epochs[-1]
    r = res[f"epoch_{ep}"][args.split]

    rl, fl = r["retain_loss"], r["forget_loss"]
    retain_name = RETAIN_LOSS_NAMES.get(cfg["retain_loss"], cfg["retain_loss"])
    forget_name = FORGET_LOSS_NAMES.get(cfg["forget_loss"], cfg["forget_loss"])
    if cfg.get("forget_ce_cap") is not None:  # bounded ascent: -min(CE, cap)
        forget_name = f"-min(CE, {cfg['forget_ce_cap']:g})"
    solver = cfg["solver"].upper()
    forget_cls = describe_forget_classes(list(cfg["forget_classes"]))

    print(
        f"{run} | epoch {ep} | {args.split} set | solver {solver} | "
        f"retain loss: {retain_name} | forget loss: {forget_name} | forgotten classes: {forget_cls}"
    )
    print(
        f"{'w_retain':>8} {'w_forget':>8} | {'retain_acc':>10} {'retain_loss':>11} | "
        f"{'forget_acc':>10} {'forget_loss':>11}"
    )
    for i, ray in enumerate(r["ray"]):
        print(
            f"{ray[0]:8.3f} {ray[1]:8.4f} | {r['retain_acc'][i]:10.4f} {rl[i]:11.4f} | "
            f"{r['forget_acc'][i]:10.4f} {fl[i]:11.4f}"
        )

    # reference model: the model before any training (epoch 0)
    z = res.get("epoch_0", {}).get(args.split)
    if z is not None:
        print(
            f"{'initial':>8} {'model':>8} | {z['retain_acc'][0]:10.4f} {z['retain_loss'][0]:11.4f} | "
            f"{z['forget_acc'][0]:10.4f} {z['forget_loss'][0]:11.4f}   (epoch 0, ~pretrained)"
        )
    else:
        print(f"initial model: not evaluated on the {args.split} split (only test at epoch 0)")

    wf = [ray[1] for ray in r["ray"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes.ravel()
    fig.suptitle(
        f"{run.name} | {args.split} set | epoch {ep} | solver {solver} | "
        f"retain loss: {retain_name} | forget loss: {forget_name} | "
        f"forgotten classes: {forget_cls}",
        fontsize=10,
    )

    # (1) Pareto front in loss space
    ax[0].plot(rl, fl, "o-", label=f"PHN, epoch {ep}")
    for i in range(len(wf)):
        ax[0].annotate(
            f"{wf[i]:.3g}", (rl[i], fl[i]), fontsize=7, xytext=(3, 3), textcoords="offset points"
        )
    ax[0].set_xlabel(f"retain loss: {retain_name} (retained classes)")
    ax[0].set_ylabel(f"forget loss: {forget_name} (forgotten classes)")
    ax[0].set_title("Front in loss space (labels = w_forget)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # (2) accuracies along the rays
    ax[1].plot(wf, r["retain_acc"], "o-", label="retained classes")
    ax[1].plot(wf, r["forget_acc"], "s-", label="forgotten classes")
    ax[1].set_xscale("symlog", linthresh=0.1)
    ax[1].set_xlabel("w_forget (symlog scale)")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy along the rays")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # (3) retain accuracy at the evaluated epochs for the first and the last ray
    # (epoch 0 has only the test split, so epochs without this split are skipped)
    eps = [e for e in epochs if args.split in res[f"epoch_{e}"]]
    for idx in (0, -1):
        name = f"ray w_forget = {wf[idx]:.3g}"
        ax[2].plot(
            eps,
            [res[f"epoch_{e}"][args.split]["retain_acc"][idx] for e in eps],
            "o-",
            label=name,
        )
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("accuracy on retained classes")
    ax[2].set_title("Retain accuracy during training")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    # (4) training objectives: mean over the steps of each epoch (each step uses a random ray,
    # so the curves depend on the ray distribution); the two objectives have different scales,
    # hence one y axis each
    train_eps = range(1, len(train_log["retain_loss"]) + 1)
    ax[3].plot(train_eps, train_log["retain_loss"], color="tab:blue", label="retain loss")
    ax[3].set_xlabel("epoch")
    ax[3].set_ylabel(f"retain loss: {retain_name}", color="tab:blue")
    ax[3].tick_params(axis="y", labelcolor="tab:blue")
    ax_f = ax[3].twinx()
    ax_f.plot(train_eps, train_log["forget_loss"], color="tab:orange", label="forget loss")
    ax_f.set_ylabel(f"forget loss: {forget_name}", color="tab:orange")
    ax_f.tick_params(axis="y", labelcolor="tab:orange")
    ax[3].set_title("Training objectives (mean over the epoch's random rays)")
    ax[3].grid(alpha=0.3)

    out = run / f"front_{args.split}_epoch{ep}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()