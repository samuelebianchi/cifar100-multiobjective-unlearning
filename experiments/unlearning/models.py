"""Hypernetwork + functional ResNet18 target for CIFAR-100 unlearning.

Adapted from experiments/multimnist/models.py (Navon et al., ICLR 2021,
AvivNavon/pareto-hypernetworks, MIT license). Main differences w.r.t. the original:

* the list of generated tensors (``layer_to_shape``) is derived from a reference
  torchvision ResNet18 instead of being hard-coded;
* 3-channel CIFAR input, optional CIFAR stem (3x3 conv, no max-pool);
* a single 100-way classification head (the original has two 10-way heads);
* the hypernetwork forward pass is vectorised over chunks (same output, much faster);
* ``num_chunks`` is computed automatically from the target size.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn


# --------------------------------------------------------------------------- #
# Reference (standard nn.Module) ResNet18 - used for pre-training and to read
# parameter names / shapes
# --------------------------------------------------------------------------- #
def build_reference_resnet18(num_classes: int = 100, cifar_stem: bool = True) -> nn.Module:
    net = models.resnet18(num_classes=num_classes)
    if cifar_stem:
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
    return net


def get_layer_to_shape(num_classes: int = 100, cifar_stem: bool = True) -> Dict[str, torch.Size]:
    """Names are prefixed with 'resnet.' to match the original code."""
    ref = build_reference_resnet18(num_classes, cifar_stem)
    return {f"resnet.{n}": p.shape for n, p in ref.named_parameters()}


def chunks_for(n_params: int, num_ws: int, w_dim: int) -> int:
    return math.ceil(n_params / (num_ws * w_dim))


# --------------------------------------------------------------------------- #
# Hypernetwork
# --------------------------------------------------------------------------- #
class ResnetHyper(nn.Module):
    def __init__(
        self,
        layer_to_shape: Dict[str, torch.Size],
        preference_dim: int = 2,
        preference_embedding_dim: int = 32,
        hidden_dim: int = 100,
        num_chunks: Optional[int] = None,
        chunk_embedding_dim: int = 64,
        num_ws: int = 11,
        w_dim: int = 10000,
        w_init_std: float = 0.1,
    ):
        """
        :param layer_to_shape: name -> shape of every tensor the target needs
        :param preference_dim: number of objectives (here: retain, forget)
        :param num_chunks: chunks fed to the MLP; if None, the minimum needed
        :param num_ws: number of W matrices (controls hypernet size, see paper)
        :param w_dim: output dim of each W matrix row block
        :param w_init_std: std of W init (original: 0.1; use much smaller in delta mode)
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

    def forward(self, preference: torch.Tensor) -> Dict[str, torch.Tensor]:
        # preference: (preference_dim,) -> weighted sum of per-objective embeddings
        pref_emb = preference @ self.preference_embedding_matrix.weight  # (pref_emb_dim,)
        chunk_emb = self.chunk_embedding_matrix.weight  # (C, chunk_emb_dim)
        inp = torch.cat([pref_emb.unsqueeze(0).expand(self.num_chunks, -1), chunk_emb], dim=1)
        rep = self.fc(inp)  # (C, hidden)
        # (C, num_ws * w_dim) -> flat, chunk-major (same ordering as the original loop)
        flat = torch.cat([F.linear(rep, w) for w in self.ws], dim=1).reshape(-1)

        out, pos = {}, 0
        for name, shape in self.layer_to_shape.items():
            n = shape.numel()
            out[name] = flat[pos : pos + n].reshape(shape)
            pos += n
        return out


# --------------------------------------------------------------------------- #
# Functional target network
# --------------------------------------------------------------------------- #
class ResNetTarget(nn.Module):
    """ResNet18 whose weights are passed in at call time (generated by the hypernet).

    BatchNorm always uses batch statistics (no running stats), as in the original code.
    => always evaluate on *mixed* batches (all classes), never on forget-only batches.
    """

    def __init__(self, cifar_stem: bool = True):
        super().__init__()
        self.cifar_stem = cifar_stem

    @staticmethod
    def _bn(x, weights, name):
        return F.batch_norm(
            x, None, None, weights[f"{name}.weight"], weights[f"{name}.bias"], training=True
        )

    def forward(self, x, weights):
        x = self.forward_init(x, weights)
        for layer in (1, 2, 3, 4):
            x = self.forward_block(x, weights, layer, 0)
            x = self.forward_block(x, weights, layer, 1)
        x = torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)
        return F.linear(x, weights["resnet.fc.weight"], weights["resnet.fc.bias"])

    def forward_init(self, x, weights):
        if self.cifar_stem:
            x = F.conv2d(x, weights["resnet.conv1.weight"], stride=1, padding=1)
            x = F.relu(self._bn(x, weights, "resnet.bn1"))
        else:
            x = F.conv2d(x, weights["resnet.conv1.weight"], stride=2, padding=3)
            x = F.relu(self._bn(x, weights, "resnet.bn1"))
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    def forward_block(self, x, weights, layer, index):
        stride = 2 if (layer > 1 and index == 0) else 1
        p = f"resnet.layer{layer}.{index}"
        identity = x
        out = F.conv2d(x, weights[f"{p}.conv1.weight"], stride=stride, padding=1)
        out = F.relu(self._bn(out, weights, f"{p}.bn1"))
        out = F.conv2d(out, weights[f"{p}.conv2.weight"], stride=1, padding=1)
        out = self._bn(out, weights, f"{p}.bn2")
        if layer > 1 and index == 0:
            identity = F.conv2d(x, weights[f"{p}.downsample.0.weight"], stride=2)
            identity = self._bn(identity, weights, f"{p}.downsample.1")
        return F.relu(out + identity)
