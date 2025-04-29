from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from models import (
    JEPAWorldModel,
    JEPAWorldModelV1,
    JEPAWorldModelV2,
    JEPAWorldModelV3,
    JEPAWorldModelV4,
)
from training import train_model


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_data(device):
    """
    Load datasets for JEPA training and probing evaluation.
    Returns:
        train_ds: for representation learning
        probe_train_ds: for prober training
        probe_val_ds: dict of validation sets
    """
    data_path = "/content/drive/My Drive/DL25SP"

    # JEPA representation training data (full trajectories)
    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
    )

    # Probing training data (predictions)
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    # Probing validation sets
    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )
    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )
    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }
    return train_ds, probe_train_ds, probe_val_ds


def load_model(device):
    # Hyperparameters
    input_channels = 1   # adjust if your data has more channels
    height = 64          # adjust to your state height
    width = 64           # adjust to your state width
    action_dim = 2
    repr_dim = 256
    hidden_size = 512

    model = JEPAWorldModel(
        input_channels=input_channels,
        height=height,
        width=width,
        action_dim=action_dim,
        repr_dim=repr_dim,
        hidden_size=hidden_size,
    )
    return model.to(device)


'''
def train_jepa_model(
    device,
    model,
    train_loader,
    epochs: int = 50,
    lr: float = 1e-3,
):
    """
    Train the JEPA model to predict future embeddings by minimizing MSE to target encodings.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"JEPA Train Epoch {epoch}"):
            states = batch.states.to(device) 
            actions = batch.actions.to(device) 

            preds = model(states=states, actions=actions)

            # Compute target embeddings via frozen encoder
            B, T, C, H, W = states.shape
            flat = states.reshape(B * T, C, H, W)
            if hasattr(model, "target_encoder") and model.target_encoder is not None:
                with torch.no_grad():
                    zt = model.target_encoder(flat)
            else:
                with torch.no_grad():
                    zt = model.encoder(flat)
            target_emb = zt.view(B, T, -1)

            loss = model.compute_jepa_loss(states, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | JEPA train MSE: {avg_loss:.4f}")
    model.eval()
''''

def evaluate_model(
    device,
    model,
    probe_train_ds,
    probe_val_ds,
):
    """
    Train a prober on model predictions and evaluate on validation sets.
    """
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    train_ds, probe_train_ds, probe_val_ds = load_data(device)
    model = load_model(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    train_model(device, model, train_ds, lr=1e-3, epochs=100)
    
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
    
    '''

    # Detect channel count
    sample_batch = next(iter(train_ds if hasattr(train_ds, 'states') else probe_train_ds))
    input_channels = sample_batch.states.shape[2]
    print(f"Detected input channels: {input_channels}")

    # Choose which models to benchmark
    MODEL_VARIANTS = [
        ("RNN_JEPA", JEPAWorldModel),
        #("MomentumJEPA_V1", JEPAWorldModelV1),
        #("VICRegJEPA_V2", JEPAWorldModelV2),
        #("ResNetJEPA_V3", JEPAWorldModelV3),
        #("V4", JEPAWorldModelV4),
    ]

    for name, cls in MODEL_VARIANTS:
        print(f"\n=== {name} ===")
        model = cls(input_channels=input_channels).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name} | Params: {total_params:,}")

        # 1) Representation learning
        train_jepa_model(device, model, train_ds, epochs=500, lr=1e-3)
        # 2) Probing evaluation
        evaluate_model(device, model, probe_train_ds, probe_val_ds)
        '''
