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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            preds = model(states, actions)
            # Compute target embeddings via encoder
            B, T1, C, H, W = states.shape
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