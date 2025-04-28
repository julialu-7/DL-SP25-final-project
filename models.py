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
        # optional target encoder (same as encoder) for possible BYOL-style
        if use_target_encoder:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_encoder = None

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # initial embedding
        s = self.encoder(states[:, 0])  # [B, repr_dim]
        preds = [s]
        # autoregressive unroll
        for t in range(actions.size(1)):
            inp = torch.cat([s, actions[:, t]], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        # return [B, T+1, D]
        return torch.stack(preds, dim=1)

    def compute_jepa_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Forward to get embeddings [B, T, D]
        preds = self(states, actions)
        # Drop initial embedding, compare only future steps
        pred_future = preds[:, 1:, :]  # [B, T-1, D]
        B, T, C, H, W = states.shape
        # Flatten frames 1..T-1
        frames = states[:, 1:, :, :, :].reshape(B * (T - 1), C, H, W)
        with torch.no_grad():
            target_feats = self.encoder(frames)
        target = target_feats.view(B, T - 1, -1)  # [B, T-1, D]
        loss = F.mse_loss(pred_future, target)
        return loss



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
        # encoder and target
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(64, repr_dim), nn.ReLU(True)
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False
        self.predictor = build_mlp([repr_dim + action_dim] + hidden_dims + [repr_dim])

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k.data = self.momentum * k.data + (1 - self.momentum) * q.data

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        s = self.encoder(states[:, 0])
        preds = [s]
        for t in range(actions.size(1)):
            inp = torch.cat([s, actions[:, t]], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        if self.training:
            self._momentum_update()
        return torch.stack(preds, dim=1)

    def compute_jepa_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        preds = self(states, actions)
        pred_future = preds[:, 1:, :]  # [B, T-1, D]
        B, T, C, H, W = states.shape
        frames = states[:, 1:, :, :, :].reshape(B * (T - 1), C, H, W)
        with torch.no_grad():
            target_feats = self.target_encoder(frames)
        target = target_feats.view(B, T - 1, -1)
        loss = F.mse_loss(pred_future, target)
        if self.training:
            self._momentum_update()
        return loss


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
        self.predictor = build_mlp([repr_dim + action_dim] + hidden_dims + [repr_dim])

    def _vicreg_losses(self, z, z_target):
        inv = F.mse_loss(z, z_target)
        std_z = torch.sqrt(z.var(0) + 1e-4)
        var = torch.mean(F.relu(1 - std_z))
        z_norm = z - z.mean(0)
        cov = (z_norm.T @ z_norm) / (z.shape[0] - 1)
        off = cov.flatten()[1:].view(self.repr_dim - 1, self.repr_dim + 1)[:, :-1]
        cov_loss = (off**2).sum() / self.repr_dim
        return inv, var * self.var_weight, cov_loss * self.cov_weight

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        s = self.encoder(states[:, 0])
        preds = [s]
        for t in range(actions.size(1)):
            inp = torch.cat([s, actions[:, t]], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        return torch.stack(preds, dim=1)

    def compute_jepa_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        preds = self(states, actions)[:, 1:, :]  # [B, T-1, D]
        B, T, C, H, W = states.shape
        frames = states[:, 1:, :, :, :].reshape(B * (T - 1), C, H, W)
        with torch.no_grad():
            zt = self.encoder(frames)
        target = zt.view(B, T - 1, -1)
        flat_pred = preds.reshape(B * (T - 1), -1)
        flat_tgt = target.reshape(B * (T - 1), -1)
        inv, var, cov_loss = self._vicreg_losses(flat_pred, flat_tgt)
        return inv + var + cov_loss

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
            inp = torch.cat([s, actions[:, t]], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        return torch.stack(preds, dim=1)

    def compute_jepa_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        preds = self(states, actions)[:, 1:, :]
        B, T, C, H, W = states.shape
        frames = states[:, 1:, :, :, :].reshape(B * (T - 1), C, H, W)
        with torch.no_grad(): target_feats = self.encoder(frames)
        target = target_feats.view(B, T - 1, -1)
        loss = F.mse_loss(preds, target)
        return loss



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
            inp = torch.cat([s, actions[:, t]], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        if self.training:
            self._momentum_update()
        return torch.stack(preds, dim=1)

    def compute_jepa_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        preds = self(states, actions)[:, 1:, :]
        B, T, C, H, W = states.shape
        frames = states[:, 1:, :, :, :].reshape(B * T, C, H, W)
        with torch.no_grad():
            zt = self.encoder(frames)
        target = zt.view(B, T, -1)
        loss = F.mse_loss(preds, target)
        # momentum update for target encoder
        if self.training:
            self._momentum_update()
        return loss


class Prober(nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch
        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1]))
        self.prober = nn.Sequential(*layers)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.prober(e)
