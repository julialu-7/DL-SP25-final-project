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
