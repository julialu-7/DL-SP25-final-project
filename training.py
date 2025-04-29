import torch
from torch import nn
from tqdm.auto import tqdm
from models import build_mlp

def train_model(device, model, train_loader, lr: float, epochs: int):
    """
    Train JEPAWorldModel using only the training data loader.
    Args:
        device: torch device
        model: JEPAWorldModel instance
        train_loader: DataLoader for representation learning (full trajectories)
        lr: learning rate
        epochs: number of epochs
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Dynamic adjustment of embed input dimension based on first batch
    batch0 = next(iter(train_loader))
    states0 = batch0.states.to(device)
    B0, T1, C, H, W = states0.shape
    feat_dim = C * H * W
    if hasattr(model, 'input_dim') and getattr(model, 'input_dim', None) != feat_dim:
        print(f"Adjusting model.embed input dim from {model.input_dim} to {feat_dim}")
        model.input_dim = feat_dim
        model.embed = build_mlp([feat_dim, 512, model.repr_dim]).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"JEPA Train Epoch {epoch}"):
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            preds = model(states, actions)
            B, T1, _, _, _ = states.shape
            flat = states.reshape(B * T1, -1)
            with torch.no_grad():
                target = model.embed(flat).view(B, T1, -1)

            loss = criterion(preds, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs}  Train MSE: {avg_loss:.4f}")