from typing import Optional

import torch
from torch import nn


ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def get_activation(name: str) -> nn.Module:
    name = name.strip().lower()
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported activation: {name}. Supported activations: {ACTIVATION_REGISTRY.keys()}"
        )
    return ACTIVATION_REGISTRY[name]


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "silu",
        dropout: float = 0.0,
        norm: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        activation_layer = get_activation(activation)
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        layers = []
        for idx, (src_dim, dst_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(src_dim, dst_dim))
            is_last = idx == len(dims) - 2
            if not is_last:
                if norm:
                    layers.append(nn.LayerNorm(dst_dim))
                layers.append(activation_layer())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0,
        norm: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()
        self.mlp = MLP(
            in_dim=dim,
            out_dim=dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            dropout=dropout,
            norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
