from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class JEPAWorldModel(torch.nn.Module):
    """
    JEPA world model: encodes frames and predicts future embeddings given actions.

    Args:
        input_channels: number of image channels (e.g. 1 or 3)
        repr_dim: dimension of representation space
        action_dim: dimensionality of actions (e.g. 2 for (dx, dy))
        hidden_dims: MLP hidden layer dims for predictor
        use_target_encoder: if True, maintain a frozen copy of encoder for target embeddings
    """
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

        # encoder: conv backbone -> global pooling -> project to repr_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, repr_dim),
            nn.ReLU(True),
        )

        # predictor: MLP from [repr_dim + action_dim] to repr_dim
        mlp_dims = [repr_dim + action_dim] + hidden_dims + [repr_dim]
        self.predictor = build_mlp(mlp_dims)

        # optional target encoder (frozen) for computing targets
        if use_target_encoder:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_encoder = None


    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict a sequence of representations.

        Args:
            states: [B, T, C, H, W] sequence of observations
            actions: [B, T-1, action_dim] sequence of actions

        Returns:
            preds: [B, T, repr_dim] predicted embeddings (s_0 ... s_{T-1})
        """
        B, T, C, H, W = states.shape
        device = states.device

        # initial representation from encoder on first frame
        # states[:, 0, ...] -> [B, C, H, W]
        s_pred = self.encoder(states[:, 0])  # [B, repr_dim]
        preds = [s_pred]

        # recurrently predict next embeddings
        for t in range(1, T):
            # action at previous timestep: [B, action_dim]
            a_prev = actions[:, t - 1]
            # concatenate state and action
            inp = torch.cat([s_pred, a_prev], dim=1)
            s_pred = self.predictor(inp)
            preds.append(s_pred)

        # stack into [B, T, repr_dim]
        preds = torch.stack(preds, dim=1)
        return preds

# ------------------------------------------------------
# Option 1: Momentum‐JEPA (BYOL‐Style)
#   - Frozen target encoder updated via momentum
#   - Predict MLP as before
# ------------------------------------------------------
class JEPAWorldModelV1(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 repr_dim: int = 256,
                 action_dim: int = 2,
                 hidden_dims: List[int] = [512, 256],
                 momentum: float = 0.99):
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
        # predictor
        self.predictor = build_mlp([repr_dim + action_dim] + hidden_dims + [repr_dim])
        # momentum target encoder
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k.data = self.momentum * k.data + (1 - self.momentum) * q.data

    def forward(self, states, actions):
        B, T, C, H, W = states.shape
        # initial
        s = self.encoder(states[:,0])
        preds = [s]
        # recurrent
        for t in range(1, T):
            inp = torch.cat([s, actions[:,t-1]], dim=1)
            s = self.predictor(inp)
            preds.append(s)
        preds = torch.stack(preds, 1)
        # update momentum
        if self.training:
            self._momentum_update()
        return preds


# ------------------------------------------------------
# Option 2: VICReg‐JEPA
#   - Add variance & covariance loss hooks in model
#   - Use same frozen target encoder (no predictor momentum)
# ------------------------------------------------------
class JEPAWorldModelV2(nn.Module):
    def __init__(self,
                 input_channels=1,
                 repr_dim=256,
                 action_dim=2,
                 hidden_dims=[256],
                 var_weight=25.0,
                 cov_weight=1.0):
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        # encoder + target
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(128, repr_dim)
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False
        self.predictor = build_mlp([repr_dim + action_dim] + hidden_dims + [repr_dim])

    def _vicreg_losses(self, z, z_target):
        # invariance (MSE)
        inv_loss = F.mse_loss(z, z_target)
        # variance
        std_z = torch.sqrt(z.var(0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_z))
        # covariance
        z_norm = z - z.mean(0)
        cov = (z_norm.T @ z_norm) / (z.shape[0] - 1)
        off_diag = cov.flatten()[1:].view(self.repr_dim-1, self.repr_dim+1)[:, :-1]
        cov_loss = (off_diag**2).sum() / self.repr_dim
        return inv_loss, var_loss * self.var_weight, cov_loss * self.cov_weight

    def forward(self, states, actions):
        B, T, C, H, W = states.shape
        s = self.encoder(states[:,0])  # [B, repr]
        preds, losses = [s], []
        for t in range(1, T):
            inp = torch.cat([s, actions[:,t-1]], 1)
            s = self.predictor(inp)
            preds.append(s)
            with torch.no_grad():
                s_tgt = self.target_encoder(states[:,t])
            losses.append(self._vicreg_losses(s, s_tgt))
        preds = torch.stack(preds, 1)
        # aggregate vicreg terms over all steps
        vic_losses = list(zip(*losses)) if losses else ([],[],[])
        return preds, vic_losses


# ------------------------------------------------------
# Option 3: Residual ResNet‐JEPA + Dropout
#   - Small ResBlock conv backbone
#   - Dropout in predictor to regularize
# ------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class JEPAWorldModelV3(nn.Module):
    def __init__(self,
                 input_channels=1,
                 repr_dim=256,
                 action_dim=2,
                 hidden_dims=[256],
                 dropout=0.2):
        super().__init__()
        self.repr_dim, self.action_dim = repr_dim, action_dim
        # ResNet backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            ResBlock(32), nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            ResBlock(64), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(64, repr_dim)
        )
        # predictor with dropout
        dims = [repr_dim + action_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(True), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], repr_dim))
        self.predictor = nn.Sequential(*layers)

    def forward(self, states, actions):
        B, T, C, H, W = states.shape
        s = self.encoder(states[:,0])
        preds = [s]
        for t in range(1, T):
            inp = torch.cat([s, actions[:,t-1]], 1)
            s = self.predictor(inp)
            preds.append(s)
        return torch.stack(preds, 1)

class Prober(torch.nn.Module):
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
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
