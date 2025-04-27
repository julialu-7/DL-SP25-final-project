from typing import List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import copy


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class JEPAWorldModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        repr_dim: int = 256,
        action_dim: int = 2,
        hidden_dims: List[int] = [256],
        use_target_encoder: bool = True,
    ):
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(128, repr_dim), nn.ReLU(True)
        )
        # predictor MLP
        self.predictor = build_mlp([repr_dim + action_dim] + hidden_dims + [repr_dim])
        # optional target encoder
        if use_target_encoder:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_encoder = None

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states: [B, T, C, H, W] but use only first frame
        # actions: [B, T, action_dim]
        # returns preds: [B, T+1, repr_dim]
        B = states.size(0)
        # encode initial frame
        s = self.encoder(states[:, 0])  # [B, repr_dim]
        preds = [s]
        # autoregressive unroll for each action step
        for t in range(actions.size(1)):
            a = actions[:, t]
            inp = torch.cat([s, a], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        return torch.stack(preds, dim=1)


class JEPAWorldModelV1(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        repr_dim: int = 256,
        action_dim: int = 2,
        hidden_dims: List[int] = [512, 256],
        momentum: float = 0.99,
    ):
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.momentum = momentum
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(64, repr_dim), nn.ReLU(True)
        )
        self.predictor = build_mlp([repr_dim + action_dim] + hidden_dims + [repr_dim])
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k.data = self.momentum * k.data + (1 - self.momentum) * q.data

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # autoregressive forward using only first frame
        s = self.encoder(states[:, 0])
        preds = [s]
        for t in range(actions.size(1)):
            a = actions[:, t]
            inp = torch.cat([s, a], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        if self.training:
            self._momentum_update()
        return torch.stack(preds, dim=1)


class JEPAWorldModelV2(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        repr_dim: int = 256,
        action_dim: int = 2,
        hidden_dims: List[int] = [256],
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
    ):
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(128, repr_dim)
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = build_mlp([repr_dim + action_dim] + hidden_dims + [repr_dim])

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        s = self.encoder(states[:, 0])
        preds = [s]
        for t in range(actions.size(1)):
            a = actions[:, t]
            inp = torch.cat([s, a], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        return torch.stack(preds, dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class JEPAWorldModelV3(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        repr_dim: int = 256,
        action_dim: int = 2,
        hidden_dims: List[int] = [256],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            ResBlock(32), nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            ResBlock(64), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(64, repr_dim)
        )
        dims = [repr_dim + action_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], repr_dim))
        self.predictor = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        s = self.encoder(states[:, 0])
        preds = [s]
        for t in range(actions.size(1)):
            a = actions[:, t]
            inp = torch.cat([s, a], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        return torch.stack(preds, dim=1)


class DeepResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch*2, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ch*2)
        self.conv2 = nn.Conv2d(ch*2, ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class JEPAWorldModelV4(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        repr_dim: int = 512,
        action_dim: int = 2,
        hidden_dims: List[int] = [512, 256],
        momentum: float = 0.99,
    ):
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.momentum = momentum
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(True),
            DeepResBlock(64), DeepResBlock(64),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(64, repr_dim)
        )
        self.predictor = build_mlp([repr_dim + action_dim] + hidden_dims + [repr_dim])
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k.data = self.momentum * k.data + (1 - self.momentum) * q.data

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        s = self.encoder(states[:, 0])
        preds = [s]
        for t in range(actions.size(1)):
            a = actions[:, t]
            inp = torch.cat([s, a], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        if self.training:
            self._momentum_update()
        return torch.stack(preds, dim=1)