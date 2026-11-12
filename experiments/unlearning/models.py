"""Hypernetwork + torchvision ResNet18 target for CIFAR-100 unlearning.

Adapted from experiments/multimnist/models.py (Navon et al., ICLR 2021,
AvivNavon/pareto-hypernetworks, MIT license).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torchvision import models


# --------------------------------------------------------------------------- #
# Reference (standard nn.Module) ResNet18 - used for pre-training and to read
# parameter names / shapes
# --------------------------------------------------------------------------- #
def build_reference_resnet18(num_classes: int = 100) -> nn.Module:
    """Build a standard torchvision ResNet18 with a CIFAR stem.

    The 7x7/stride-2 convolution and the max-pool are replaced by a 3x3/stride-1 convolution
    (better suited to 32x32 inputs). Used for pre-training and as the source of parameter
    names and shapes.

    Args:
        num_classes: Number of output classes.

    Returns:
        A randomly initialised ResNet18.
    """
    net = models.resnet18(num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


def get_layer_to_shape(num_classes: int = 100) -> dict[str, torch.Size]:
    """Map every parameter name of the target ResNet18 to its shape.

    Names are those of ``named_parameters()`` of the torchvision model.

    Args:
        num_classes: Number of output classes.

    Returns:
        Ordered dictionary ``{param name: shape}``.
    """
    ref = build_reference_resnet18(num_classes)
    return {n: p.shape for n, p in ref.named_parameters()}


def chunks_for(n_params: int, num_ws: int, w_dim: int) -> int:
    """Number of chunks the hypernetwork needs to generate ``n_params`` weights.

    Args:
        n_params: Total number of parameters of the target network.
        num_ws: Number of W matrices of the hypernetwork.
        w_dim: Output dimension of each W matrix.

    Returns:
        Smallest number of chunks whose total capacity is at least ``n_params``.
    """
    return math.ceil(n_params / (num_ws * w_dim))


# --------------------------------------------------------------------------- #
# Hypernetwork
# --------------------------------------------------------------------------- #
class ResnetHyper(nn.Module):
    """Hypernetwork that maps a preference ray to the flat weights of the target ResNet18."""

    def __init__(
        self,
        layer_to_shape: dict[str, torch.Size],
        preference_dim: int = 2,
        preference_embedding_dim: int = 32,
        hidden_dim: int = 100,
        num_chunks: int | None = None,
        chunk_embedding_dim: int = 64,
        num_ws: int = 11,
        w_dim: int = 10000,
        w_init_std: float = 0.1,
    ) -> None:
        """Initialise the hypernetwork.

        Args:
            layer_to_shape: Name -> shape of every tensor the target network needs.
            preference_dim: Number of objectives (here: retain, forget).
            preference_embedding_dim: Size of the preference embedding.
            hidden_dim: Width of the hidden layers of the MLP.
            num_chunks: Chunks fed to the MLP; if None, the minimum number needed.
            chunk_embedding_dim: Size of the chunk embedding.
            num_ws: Number of W matrices.
            w_dim: Output dimension of each W matrix row block.
            w_init_std: Std of the W initialisation

        Raises:
            AssertionError: If the capacity of the hypernetwork is smaller than the number
                of target parameters.
        """
        super().__init__()
        self.layer_to_shape = layer_to_shape
        n_target = sum(s.numel() for s in layer_to_shape.values())
        if num_chunks is None:
            num_chunks = chunks_for(n_target, num_ws, w_dim)
        assert num_chunks * num_ws * w_dim >= n_target, (
            f"hypernet capacity {num_chunks * num_ws * w_dim} < target params {n_target}"
        )
        self.num_chunks = num_chunks
        self.preference_embedding_dim = preference_embedding_dim

        self.chunk_embedding_matrix = nn.Embedding(num_chunks, chunk_embedding_dim)
        self.preference_embedding_matrix = nn.Embedding(preference_dim, preference_embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(preference_embedding_dim + chunk_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ws = nn.ParameterList(
            [nn.Parameter(torch.randn(w_dim, hidden_dim)) for _ in range(num_ws)]
        )

        nn.init.normal_(self.preference_embedding_matrix.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.chunk_embedding_matrix.weight, mean=0.0, std=0.1)
        for w in self.ws:
            nn.init.normal_(w, mean=0.0, std=w_init_std)

    def forward(self, preference: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate the target weights for a given preference ray.

        Args:
            preference: Preference ray, shape ``(preference_dim,)``.

        Returns:
            Dictionary mapping each parameter name to its generated tensor.
        """
        # preference: (preference_dim,) -> weighted sum of per-objective embeddings
        pref_emb = preference @ self.preference_embedding_matrix.weight  # (pref_emb_dim,)
        chunk_emb = self.chunk_embedding_matrix.weight  # (C, chunk_emb_dim)
        inp = torch.cat([pref_emb.unsqueeze(0).expand(self.num_chunks, -1), chunk_emb], dim=1)
        rep = self.fc(inp)  # (C, hidden)
        # (C, num_ws * w_dim)
        flat = torch.cat([F.linear(rep, w) for w in self.ws], dim=1).reshape(-1)

        out, pos = {}, 0
        for name, shape in self.layer_to_shape.items():
            n = shape.numel()
            out[name] = flat[pos : pos + n].reshape(shape)
            pos += n
        return out


# --------------------------------------------------------------------------- #
# Target network: the torchvision ResNet18 itself, called with external weights
# --------------------------------------------------------------------------- #
def forward_with_weights(
    net: nn.Module, x: torch.Tensor, weights: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Run a standard ``nn.Module`` using the given weights instead of its own parameters.

    The parameters named in ``weights`` (e.g. generated by the hypernetwork) replace the
    module's ones for this call only, and gradients flow back to them. Everything that is not
    in ``weights`` (BatchNorm buffers) is taken from the module. BatchNorm follows the
    mode of the module: batch statistics in ``train()``, the stored running statistics in
    ``eval()``.

    Args:
        net: The target network (e.g. the output of `build_reference_resnet18`).
        x: Input images, shape ``(B, 3, H, W)``.
        weights: Parameter name -> tensor produced by the hypernetwork.

    Returns:
        Logits, shape ``(B, num_classes)``.
    """
    return functional_call(net, weights, (x,))
