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
    """
    Joint Embedding Prediction Architecture (JEPA) world model.
    Uses an MLP embedder, an RNN for temporal dynamics, and a linear spatial predictor.
    Supports both training (teacher-forced) and inference modes.

    Input:
        states: [B, T+1, C, H, W] (T+1 > 1 for training, T+1 == 1 for inference)
        actions: [B, T, action_dim]
    Output:
        preds_full: [B, T+1, repr_dim]
    """
    def __init__(
        self,
        input_channels: int,
        height: int,
        width: int,
        action_dim: int = 2,
        repr_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()
        self.repr_dim = repr_dim
        self.input_dim = input_channels * height * width
        self.action_dim = action_dim

        # Embed raw states to a compact representation
        self.embed = build_mlp([self.input_dim, 512, repr_dim])

        # RNN to capture temporal dynamics
        self.rnn = nn.LSTM(
            input_size=repr_dim + action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Spatial predictor: from hidden state to next-step representation
        self.predictor = nn.Linear(hidden_size, repr_dim)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, T1, C, H, W = states.shape
        # Flatten states batch+time for embedding
        if T1 == 1:
            # Inference mode: only initial state provided
            # Embed initial state
            flat_init = states.view(B, -1)
            embed_init = self.embed(flat_init)  # [B, repr_dim]
            embed_init = embed_init.view(B, 1, self.repr_dim)  # [B, 1, repr_dim]

            # Prepare repeated features for RNN input
            T = actions.shape[1]
            # Repeat initial embedding for each action timestep
            embed_repeat = embed_init.repeat(1, T, 1)  # [B, T, repr_dim]
            rnn_in = torch.cat([embed_repeat, actions], dim=-1)  # [B, T, repr+action]

            # RNN forward
            rnn_out, _ = self.rnn(rnn_in)  # [B, T, hidden]
            preds_step = self.predictor(rnn_out)  # [B, T, repr_dim]

            # Concatenate initial embedding with predictions
            preds_full = torch.cat([embed_init, preds_step], dim=1)  # [B, T+1, repr_dim]
            return preds_full
        else:
            # Training mode: full states sequence available
            flat_states = states.view(B * T1, -1)
            embeds = self.embed(flat_states).view(B, T1, self.repr_dim)  # [B, T+1, repr_dim]

            # Teacher-forced RNN input: use true embeddings
            rnn_in = torch.cat([embeds[:, :-1, :], actions], dim=-1)  # [B, T, repr+action]
            rnn_out, _ = self.rnn(rnn_in)  # [B, T, hidden]
            preds_step = self.predictor(rnn_out)  # [B, T, repr_dim]

            # Prepend initial embedding
            preds_full = torch.cat([embeds[:, :1, :], preds_step], dim=1)  # [B, T+1, repr_dim]
            return preds_full


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
