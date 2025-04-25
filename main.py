from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import (
    JEPAWorldModel,
    JEPAWorldModelV1,
    JEPAWorldModelV2,
    JEPAWorldModelV3,
)


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    """
    Load training and validation dataloaders for probing.
    Returns:
        probe_train_ds, probe_val_ds dict
    """
    data_path = "/scratch/DL25SP"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

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
    return probe_train_ds, probe_val_ds


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
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
    # Load data and detect channels
    probe_train_ds, probe_val_ds = load_data(device)
    sample_batch = next(iter(probe_train_ds))
    input_channels = sample_batch.states.shape[2]
    print(f"Detected input channels: {input_channels}")

    # Define and run all model variants
    MODEL_VARIANTS = [
        ("BaseJEPA", JEPAWorldModel),
        ("MomentumJEPA_V1", JEPAWorldModelV1),
        ("VICRegJEPA_V2", JEPAWorldModelV2),
        ("ResNetJEPA_V3", JEPAWorldModelV3),
    ]

    for name, cls in MODEL_VARIANTS:
        print(f"\n=== Training & Evaluating {name} ===")
        # Instantiate with correct input channels
        model = cls(input_channels=input_channels).to(device)
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name} | Trainable Params: {total_params:,}")
        # Run probing pipeline
        evaluate_model(device, model, probe_train_ds, probe_val_ds)